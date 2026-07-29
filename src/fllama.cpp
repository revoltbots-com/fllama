// fllama.cpp — Phase 3: thin adapter over llama.cpp's server_context
//
// fllama_inference() spawns a reader thread that posts a server_task and
// streams results back via the Dart callback.  Multiple concurrent calls
// are batched automatically by server_context::update_slots().

#include "fllama.h"
#include "fllama_inference_queue.h"
#include "fllama_mtmd.h"

// server-context headers (no HTTP / httplib dependency)
#include "server-context.h"
#include "server-task.h"
#include "server-common.h"

#include "llama.cpp/common/chat.h"
#include "llama.cpp/common/common.h"
#include "llama.cpp/ggml/include/ggml.h"
#include "llama.cpp/include/llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ggml-backend.h"

// ── Logging ──────────────────────────────────────────────────────────────────

static constexpr bool kLogVerboseDebugMessages = false;

static void log_message(const char *msg,
                        fllama_log_callback logger = nullptr) {
  if (!logger) {
    fprintf(stderr, "%s\n", msg);
    fflush(stderr);
    return;
  }
  static std::mutex mtx;
  static std::deque<std::string> q;
  std::string s(msg);
  for (size_t p = 0; (p = s.find('\n', p)) != std::string::npos; p += 4)
    s.replace(p, 1, "[NL]");
  std::lock_guard<std::mutex> lk(mtx);
  q.push_back(std::move(s));
  while (q.size() > 1000)
    q.pop_front();
  logger(q.back().c_str());
}
static void log_message(const std::string &m,
                        fllama_log_callback l = nullptr) {
  log_message(m.c_str(), l);
}

static bool fllama_is_noisy_per_token_llama_log(const char *text) {
  if (!text) {
    return false;
  }
  std::string s(text);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
    s.pop_back();
  }
  return s == "set_embeddings: value = 0" ||
         s == "set_adapters_lora: adapters = 0" ||
         s == "adapters_lora_are_same: adapters = 0";
}

static void fllama_filtered_llama_log_callback(enum ggml_log_level level,
                                               const char *text,
                                               void *user_data) {
  if (fllama_is_noisy_per_token_llama_log(text)) {
    return;
  }
  common_log_default_callback(level, text, user_data);
}

struct fllama_callback_payload {
  std::string response;
  std::string openai_json;
};

static void emit_inference_callback(fllama_inference_callback callback,
                                    std::string response,
                                    std::string openai_json,
                                    uint8_t done) {
  if (!callback) {
    return;
  }

  // NativeCallable.listener relays calls from worker threads back to the Dart
  // isolate asynchronously. Pointer arguments are just addresses, so c_str()
  // into a local std::string can be dead by the time Dart reads it. Keep a
  // rolling native-side copy alive, same as log_message() does for logger
  // callbacks above.
  static std::mutex mtx;
  static std::deque<fllama_callback_payload> q;
  std::lock_guard<std::mutex> lk(mtx);
  q.push_back({std::move(response), std::move(openai_json)});
  while (q.size() > 1000) {
    q.pop_front();
  }
  const auto &payload = q.back();
  callback(payload.response.c_str(), payload.openai_json.c_str(), done);
}

// ── Globals ──────────────────────────────────────────────────────────────────

// Intentionally leaked — avoids static destruction order crash on exit.
// (ggml Metal statics may be destroyed before g_mgr's destructor runs,
//  causing ggml_abort when server_context tries to free Metal resources.)
static ServerManager &g_mgr = *new ServerManager();
static std::once_flag  g_backend_init;

static void fllama_backend_init_once() {
  std::call_once(g_backend_init, [] {
    ggml_backend_load_all();
    llama_backend_init();
    llama_log_set(fllama_filtered_llama_log_callback, nullptr);
  });
}

static std::vector<ggml_backend_dev_t> fllama_get_gpu_devices() {
  fllama_backend_init_once();

  std::vector<ggml_backend_dev_t> devices;
  for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
    auto * dev = ggml_backend_dev_get(i);
    if (dev == nullptr) {
      continue;
    }
    if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
      continue;
    }
    devices.push_back(dev);
  }
  return devices;
}

static void fllama_copy_cstr(char * dst, size_t cap, const char * src) {
  if (dst == nullptr || cap == 0) {
    return;
  }
  std::snprintf(dst, cap, "%s", src ? src : "");
}

static bool fllama_error_requires_backend_recreation(const std::string &msg) {
  // llama.cpp reports Metal command-buffer OOM / poisoned backend failures to
  // callers as a generic "Compute error.".  Once the backend is in that state,
  // the fix is to destroy and recreate the server_context rather than retrying
  // on the same context.
  return msg.find("Compute error") != std::string::npos ||
         msg.find("backend is in error state") != std::string::npos ||
         msg.find("failed to compute graph") != std::string::npos ||
         msg.find("failed to decode") != std::string::npos ||
         msg.find("OutOfMemory") != std::string::npos ||
         msg.find("out of memory") != std::string::npos;
}

// ── The actual inference logic (runs on per-request thread) ──────────────────

static void run_inference(fllama_inference_request request,
                          fllama_inference_callback callback) {
  try {
    int64_t t0 = ggml_time_ms();
    log_message("[fllama] Inference start", request.dart_logger);

    // One-time backend init.
    fllama_backend_init_once();

    // ── 1. Build common_params ────────────────────────────────────────

    common_params params;
    params.model.path       = request.model_path;
    params.n_ctx            = request.context_size;
    // Match llama.cpp server defaults more closely instead of tying batch
    // sizes to the full context window.
    params.n_batch          = std::min<int32_t>(request.context_size, 2048);
    params.n_ubatch         = std::min<int32_t>(params.n_batch, 512);
    params.flash_attn_type  = LLAMA_FLASH_ATTN_TYPE_AUTO;
    params.n_parallel       = ServerManager::DEFAULT_N_PARALLEL;
    params.n_predict        = request.max_tokens;
    params.sampling.temp    = request.temperature;
    params.sampling.top_p   = request.top_p;
    params.sampling.penalty_freq   = request.penalty_freq;
    params.sampling.penalty_repeat = request.penalty_repeat;
    params.cpuparams.n_threads     = request.num_threads;
    params.use_jinja = true;
    params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;

    // Default is 8192 MiB — way too much for mobile/embedded.
    // 0 = disable host-memory prompt caching entirely.
    // The KV cache in the llama_context still handles prompt reuse;
    // this only controls the EXTRA host-RAM cache from PR #16391.
    params.cache_ram_mib = 0;

#if TARGET_IPHONE_SIMULATOR
    params.n_gpu_layers = 0;
#else
    params.n_gpu_layers = request.num_gpu_layers;
#endif

    if (request.model_mmproj_path && strlen(request.model_mmproj_path) > 0)
      params.mmproj.path = request.model_mmproj_path;

    // Multi-Token Prediction (MTP) speculative decoding. When a drafter/
    // assistant GGUF is supplied (e.g. gemma-4-31B-it-assistant), wire it into
    // common_params.speculative; server_context handles the shared-KV draft
    // loop. Leave draft cache_type_k/v at their F16 defaults — Q8 KV cache was
    // shown to collapse MTP draft acceptance (llama.cpp#23398).
    if (request.draft_model_path && strlen(request.draft_model_path) > 0) {
      params.speculative.draft.mparams.path = request.draft_model_path;
      params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
      params.speculative.draft.n_max =
          request.draft_n_max > 0 ? request.draft_n_max : 3;
      if (request.draft_p_min >= 0.0f) {
        params.speculative.draft.p_min = request.draft_p_min;
      }
      log_message("[fllama] MTP draft enabled: n_max=" +
                      std::to_string(params.speculative.draft.n_max) +
                      ", p_min=" +
                      std::to_string(params.speculative.draft.p_min),
                  request.dart_logger);
      // Offload the (small) drafter alongside the target model.
      params.speculative.draft.n_gpu_layers = params.n_gpu_layers;
    }

    // ── 2. Get or create server_context ───────────────────────────────

    auto *srv = g_mgr.get_or_create(
        request.model_path, params, request.dart_logger);
    if (!srv || !srv->srv_ctx) {
      emit_inference_callback(
          callback, "Error: Failed to create inference context", "", true);
      return;
    }
    // RAII — release when we leave scope.
    struct Guard {
      ServerManager &m; std::string p;
      ~Guard() { m.release(p); }
    } guard{g_mgr, request.model_path};

    log_message("[fllama] Model ready (" +
                    std::to_string(ggml_time_ms() - t0) + " ms)",
                request.dart_logger);

    // ── 3. Build the prompt ───────────────────────────────────────────

    std::string prompt = request.input ? request.input : "";
    common_chat_parser_params parser_params;
    common_chat_params chat_params;
    bool is_oai = false;

    if (request.openai_request_json_string) {
      is_oai = true;
      try {
        auto body = nlohmann::ordered_json::parse(
            request.openai_request_json_string);

        std::string jinja_tmpl;
        if (body.contains("jinja_template") &&
            body["jinja_template"].is_string()) {
          jinja_tmpl = body["jinja_template"].get<std::string>();
          body.erase("jinja_template");
        }

        auto *lctx  = srv->srv_ctx->get_llama_context();
        auto *model  = llama_get_model(lctx);
        auto  tmpls  = common_chat_templates_init(model, jinja_tmpl);

        try {
          std::map<std::string, std::string> empty;
          common_chat_format_example(tmpls.get(), true, empty);
        } catch (...) {
          tmpls = common_chat_templates_init(model, "chatml");
        }

        if (body.contains("messages") && body["messages"].is_array()) {
          common_chat_templates_inputs inputs;
          inputs.use_jinja = true;
          inputs.add_generation_prompt = true;
          inputs.messages =
              common_chat_msgs_parse_oaicompat(body["messages"]);

          // Default to automatic reasoning extraction for modern reasoning/
          // channel-based templates (Qwen, GPT-OSS/Harmony, etc). Allow the
          // request body to override explicitly.
          inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
          inputs.enable_thinking = true;
          if (body.contains("reasoning_format") && body["reasoning_format"].is_string()) {
            inputs.reasoning_format = common_reasoning_format_from_name(
                body["reasoning_format"].get<std::string>());
          }

          if (body.contains("tools")) {
            inputs.tools =
                common_chat_tools_parse_oaicompat(body["tools"]);
            inputs.tool_choice =
                body.contains("tool_choice")
                    ? common_chat_tool_choice_parse_oaicompat(
                          body["tool_choice"]
                              .template get<std::string>())
                    : COMMON_CHAT_TOOL_CHOICE_AUTO;
          }

          auto result =
              common_chat_templates_apply(tmpls.get(), inputs);
          chat_params = result;
          prompt = result.prompt;
          parser_params = common_chat_parser_params(result);
          parser_params.reasoning_format = inputs.reasoning_format;
          parser_params.reasoning_in_content =
              (inputs.reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY);
          if (!result.parser.empty()) {
            parser_params.parser.load(result.parser);
          }

          if constexpr (kLogVerboseDebugMessages) {
            log_message("[fllama] fllama inputs.reasoning_format=" +
                            std::string(common_reasoning_format_name(inputs.reasoning_format)),
                        request.dart_logger);
            log_message("[fllama] Chat format: " +
                            std::string(common_chat_format_name(
                                result.format)),
                        request.dart_logger);
            log_message("[fllama] PROMPT (" +
                            std::to_string(prompt.size()) + " chars):\n" +
                            prompt,
                        request.dart_logger);
          }
        }
      } catch (const std::exception &e) {
        std::string msg = "Error: OAI parse error: " + std::string(e.what());
        log_message(std::string("[fllama] OAI parse error: ") + e.what(),
                    request.dart_logger);
        emit_inference_callback(callback, msg, "", true);
        g_mgr.clear_cancel(request.request_id);
        g_mgr.unregister_request_thread(request.request_id);
        return;
      }
    }

    // ── 4. Multimodal — extract base64 → raw bytes ───────────────────

    std::vector<raw_buffer> files;
    if (fllama_prompt_contains_image(prompt)) {
      const char * media_marker = get_media_marker();
      auto img = fllama_extract_images(prompt, media_marker);
      prompt = std::move(img.text_with_markers);
      for (auto &fb : img.file_bytes)
        files.push_back(std::move(fb));
      log_message("[fllama] Extracted " +
                      std::to_string(files.size()) + " image(s) using marker " +
                      media_marker,
                  request.dart_logger);
    }

    // ── 5. Create & post the server task ──────────────────────────────

    auto reader = srv->srv_ctx->get_response_reader();

    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.id         = reader.get_new_id();
    task.index      = 0;
    task.cli        = true;
    task.cli_prompt = prompt;
    task.cli_files  = std::move(files);

    task.params.stream       = true;
    task.params.cache_prompt = true;
    task.params.n_predict    = request.max_tokens;
    task.params.sampling.temp           = request.temperature;
    task.params.sampling.top_p          = request.top_p;
    task.params.sampling.penalty_freq   = request.penalty_freq;
    task.params.sampling.penalty_repeat = request.penalty_repeat;

    if (is_oai) {
      if (!chat_params.grammar.empty()) {
        task.params.sampling.grammar = common_grammar(
            COMMON_GRAMMAR_TYPE_TOOL_CALLS,
            chat_params.grammar);
      }
      task.params.sampling.grammar_lazy = chat_params.grammar_lazy;
      task.params.sampling.grammar_triggers = chat_params.grammar_triggers;
      task.params.sampling.generation_prompt = chat_params.generation_prompt;
      task.params.antiprompt = chat_params.additional_stops;

      auto *lctx = srv->srv_ctx->get_llama_context();
      auto *model = llama_get_model(lctx);
      auto *vocab = llama_model_get_vocab(model);
      for (const auto &preserved_token : chat_params.preserved_tokens) {
        auto ids = common_tokenize(vocab, preserved_token, false, true);
        if (ids.size() == 1) {
          task.params.sampling.preserved_tokens.insert(ids[0]);
        }
      }
    }

    // A caller-supplied GBNF grammar. The struct has carried this field, and
    // the Dart side has marshalled it across the FFI boundary, since upstream;
    // nothing ever read it here, so passing one was silently a no-op.
    //
    // Placed AFTER the is_oai block deliberately, so an explicit grammar wins
    // over whatever grammar the chat template derived for tool calls.
    //
    // COMMON_GRAMMAR_TYPE_USER is load-bearing rather than cosmetic:
    // common_grammar_needs_prefill() (common.h:204) is true only for
    // OUTPUT_FORMAT and TOOL_CALLS, and for those sampling.cpp feeds
    // generation_prompt tokens into the grammar sampler. A user GBNF's root
    // rule does not accept an assistant header, so tagging one as TOOL_CALLS
    // would throw. USER is the tag the --grammar CLI flag uses.
    if (request.grammar != nullptr && strlen(request.grammar) > 0) {
      task.params.sampling.grammar =
          common_grammar(COMMON_GRAMMAR_TYPE_USER, request.grammar);
      // A user grammar is not lazily triggered. Clear any lazy state the
      // tool-call path set above, or the sampler pairs this grammar with
      // trigger patterns that were built for a different one.
      task.params.sampling.grammar_lazy = false;
      task.params.sampling.grammar_triggers.clear();
    }

    std::random_device rd;
    task.params.sampling.seed = rd();

    if (is_oai) {
      task.params.res_type           = TASK_RESPONSE_TYPE_OAI_CHAT;
      task.params.oaicompat_model    = request.model_path;
      task.params.oaicompat_cmpl_id  = gen_chatcmplid();
      task.params.chat_parser_params = parser_params;
    }

    reader.post_task(std::move(task));

    // ── 6. Read results, invoke callbacks ─────────────────────────────

    int rid = request.request_id;
    auto should_stop = [&] { return g_mgr.is_cancelled(rid); };

    std::string full_content;
    std::string last_json;

    while (reader.has_next()) {
      server_task_result_ptr res;
      try {
        res = reader.next(should_stop);
      } catch (const std::exception &e) {
        // Final parse can fail (e.g. doubled generated_text in update_chat_msg).
        // Log and break — we still have the accumulated text + last good JSON.
        if constexpr (kLogVerboseDebugMessages) {
          log_message(std::string("[fllama] reader.next() threw: ") + e.what(),
                      request.dart_logger);
        }
        break;
      }
      if (!res) break;

      if (res->is_error()) {
        auto ej = res->to_json();
        std::string msg = ej.contains("message")
                              ? ej["message"].get<std::string>()
                              : ej.dump();
        if (fllama_error_requires_backend_recreation(msg)) {
          log_message("[fllama] Backend compute error; marking context "
                      "unhealthy so the next request recreates it",
                      request.dart_logger);
          g_mgr.mark_unhealthy(request.model_path);
        }
        emit_inference_callback(callback, msg, "", true);
        g_mgr.clear_cancel(rid);
        g_mgr.unregister_request_thread(rid);
        return;
      }

      auto *partial =
          dynamic_cast<server_task_result_cmpl_partial *>(res.get());
      if (partial) {
        full_content += partial->content;
        if constexpr (kLogVerboseDebugMessages) {
          log_message("[fllama] token: \"" + partial->content +
                      "\"  cumulative(" + std::to_string(full_content.size()) +
                      " chars)",
                      request.dart_logger);
        }
        auto j = res->to_json();
        if (!j.is_null()) {
          last_json = j.dump();
          emit_inference_callback(callback, full_content, last_json, false);
        }
        continue;
      }

      auto *final_r =
          dynamic_cast<server_task_result_cmpl_final *>(res.get());
      if (final_r) {
        // Keep accumulated full_content — final_r->content can be
        // empty or corrupted for tool-call / reasoning completions.
        try {
          auto j = res->to_json();
          last_json = j.is_null() ? "" : j.dump();
          if constexpr (kLogVerboseDebugMessages) {
            log_message("[fllama] final to_json() is_null=" +
                            std::to_string(j.is_null()) +
                            " type=" + std::to_string((int)j.type()) +
                            " size=" + std::to_string(j.size()) +
                            " dump=" + last_json.substr(0, 200),
                        request.dart_logger);
          }
        } catch (const std::exception &e) {
          if constexpr (kLogVerboseDebugMessages) {
            log_message(std::string("[fllama] final to_json() THREW: ") + e.what(),
                        request.dart_logger);
          }
          last_json = "";
        }
        if constexpr (kLogVerboseDebugMessages) {
          log_message("[fllama] final_r->content(" +
                          std::to_string(final_r->content.size()) +
                          ")=\"" + final_r->content.substr(0, 100) + "\"",
                      request.dart_logger);
        }
        emit_inference_callback(callback, full_content, last_json, true);

        log_message("[fllama] Done. " +
                        std::to_string(final_r->n_decoded) + " tok, " +
                        std::to_string(ggml_time_ms() - t0) + " ms",
                    request.dart_logger);
        g_mgr.clear_cancel(rid);
        g_mgr.unregister_request_thread(rid);
        return;
      }
    }

    // Cancelled or exhausted without final result.
    emit_inference_callback(callback, full_content, last_json, true);
    g_mgr.clear_cancel(rid);
    g_mgr.unregister_request_thread(rid);

  } catch (const std::exception &e) {
    std::string msg = "Error: " + std::string(e.what());
    emit_inference_callback(callback, msg, "", true);
    g_mgr.clear_cancel(request.request_id);
    g_mgr.unregister_request_thread(request.request_id);
  } catch (...) {
    emit_inference_callback(callback, "Error: Unknown exception", "", true);
    g_mgr.clear_cancel(request.request_id);
    g_mgr.unregister_request_thread(request.request_id);
  }
}

// ── Embedding path (BGE-M3 query encoder for Pasce Oves offline search) ──────
// Self-contained low-level embedding on its OWN model+context handle — it does
// NOT touch the server_context generation path above, so the chat model and the
// embedder never interfere. Mirrors llama.cpp/examples/embedding/embedding.cpp,
// the same recipe validated at parity against llama-server --embeddings. One
// handle per model path is cached so repeated query embeds don't reload the
// ~438 MB GGUF; guarded by a mutex (fllama_embed may be called from a Dart
// isolate).
namespace {

struct FllamaEmbedHandle {
  llama_model *model = nullptr;
  llama_context *ctx = nullptr;
  int n_embd = 0;
};

std::mutex g_embed_mutex;
std::unordered_map<std::string, FllamaEmbedHandle> g_embed_cache;

// Loads (or returns cached) an embedding handle for [model_path]. Returns
// nullptr on load failure. Caller holds g_embed_mutex.
FllamaEmbedHandle *fllama_embed_handle(const char *model_path) {
  auto it = g_embed_cache.find(model_path);
  if (it != g_embed_cache.end()) {
    return &it->second;
  }
  fllama_backend_init_once();

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = 0; // CPU: the embedder is small + latency-tolerant.
  llama_model *model = llama_model_load_from_file(model_path, mparams);
  if (model == nullptr) {
    return nullptr;
  }

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 512;
  cparams.n_ubatch = 512;
  cparams.embeddings = true;
  // UNSPECIFIED honors the GGUF's own pooling metadata (BGE-M3 = CLS), the
  // config that passed the desktop parity gate.
  cparams.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
  llama_context *ctx = llama_init_from_model(model, cparams);
  if (ctx == nullptr) {
    llama_model_free(model);
    return nullptr;
  }

  FllamaEmbedHandle h;
  h.model = model;
  h.ctx = ctx;
  h.n_embd = llama_model_n_embd(model);
  auto res = g_embed_cache.emplace(std::string(model_path), h);
  return &res.first->second;
}

} // namespace

// ── FFI entry points ─────────────────────────────────────────────────────────

extern "C" {

EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT int fllama_get_gpu_device_count(void) {
  return static_cast<int>(fllama_get_gpu_devices().size());
}

EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT int fllama_get_gpu_memory_info(
    int gpu_index,
    struct fllama_gpu_memory_info * out_info) {
  if (out_info == nullptr) {
    return 1;
  }

  std::memset(out_info, 0, sizeof(*out_info));

  if (gpu_index < 0) {
    return 2;
  }

  auto devices = fllama_get_gpu_devices();
  if (static_cast<size_t>(gpu_index) >= devices.size()) {
    return 3;
  }

  auto * dev = devices[static_cast<size_t>(gpu_index)];
  ggml_backend_dev_props props{};
  ggml_backend_dev_get_props(dev, &props);

  size_t total = props.memory_total;
  size_t free = props.memory_free;

  // Metal reports free as recommendedMaxWorkingSetSize - currentAllocatedSize.
  // If currentAllocatedSize exceeds the recommendation, the backend can
  // underflow the unsigned subtraction. Clamp that to zero here.
  if (total == 0) {
    free = 0;
  } else if (free > total) {
    free = 0;
  }

  out_info->device_index = gpu_index;
  out_info->total_bytes = static_cast<uint64_t>(total);
  out_info->free_bytes = static_cast<uint64_t>(free);
  fllama_copy_cstr(out_info->name, sizeof(out_info->name), props.name);
  fllama_copy_cstr(
      out_info->description,
      sizeof(out_info->description),
      props.description);
  fllama_copy_cstr(
      out_info->device_id,
      sizeof(out_info->device_id),
      props.device_id);
  return 0;
}

EMSCRIPTEN_KEEPALIVE void fllama_inference(fllama_inference_request request,
                                           fllama_inference_callback callback) {
  int rid = request.request_id;
  std::thread t([request, callback] { run_inference(request, callback); });
  g_mgr.register_request_thread(rid, std::move(t));
}

EMSCRIPTEN_KEEPALIVE void
fllama_inference_sync(fllama_inference_request request,
                      fllama_inference_callback callback) {
  // Synchronous variant — blocks the calling thread.
  run_inference(request, callback);
}

EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT void
fllama_inference_cancel(int request_id) {
  g_mgr.cancel(request_id);
}

// Embeds [input] with the GGUF at [model_path], writing [out_len] normalized
// floats to [out]. Returns 0 on success, negative on error:
//   -1 bad args, -2 model load failed, -3 dim mismatch (out_len != n_embd),
//   -4 empty tokenization, -5 decode failed, -6 no sequence embedding.
// out_len MUST equal the model's embedding dim (1024 for BGE-M3); the Dart
// side treats any non-zero return as a typed failure and falls back to BM25.
EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT int
fllama_embed(const char *model_path, const char *input, float *out,
             int out_len) {
  if (model_path == nullptr || input == nullptr || out == nullptr ||
      out_len <= 0) {
    return -1;
  }
  std::lock_guard<std::mutex> lock(g_embed_mutex);
  FllamaEmbedHandle *h = fllama_embed_handle(model_path);
  if (h == nullptr) {
    return -2;
  }
  if (h->n_embd != out_len) {
    return -3;
  }

  const llama_vocab *vocab = llama_model_get_vocab(h->model);
  std::vector<llama_token> tokens =
      common_tokenize(vocab, std::string(input), /*add_special=*/true,
                      /*parse_special=*/true);
  if (tokens.empty()) {
    return -4;
  }
  if (static_cast<int>(tokens.size()) > 512) {
    tokens.resize(512);
  }

  // Fresh KV each call — every query is independent.
  llama_memory_clear(llama_get_memory(h->ctx), true);
  llama_batch batch =
      llama_batch_init(static_cast<int>(tokens.size()), 0, 1);
  for (int i = 0; i < static_cast<int>(tokens.size()); i++) {
    common_batch_add(batch, tokens[i], i, {0}, /*logits=*/true);
  }
  if (llama_decode(h->ctx, batch) < 0) {
    llama_batch_free(batch);
    return -5;
  }
  const float *embd = llama_get_embeddings_seq(h->ctx, 0);
  if (embd == nullptr) {
    llama_batch_free(batch);
    return -6;
  }
  // embd_norm = 2 -> L2 normalization, matching the pack + query convention.
  common_embd_normalize(embd, out, h->n_embd, /*embd_norm=*/2);
  llama_batch_free(batch);
  return 0;
}

} // extern "C"
