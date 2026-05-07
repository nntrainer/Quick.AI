// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    openai_server.cpp
 * @brief   OpenAI-compatible REST server for Quick.AI causal LM models.
 */

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <codecvt>
#include <locale>
#include <winsock2.h>
#include <ws2tcpip.h>
using socket_t = SOCKET;
static constexpr socket_t invalid_socket_value = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
static constexpr socket_t invalid_socket_value = -1;
#endif

#include "causal_lm.h"
#include "chat_template.h"
#include "embedding_gemma.h"
#include "factory.h"
#include "gemma3_causallm.h"
#include "gptoss_cached_slim_causallm.h"
#include "gptoss_causallm.h"
#include "json.hpp"
#include "openai_protocol.h"
#include "qwen2_causallm.h"
#include "qwen2_embedding.h"
#include "qwen3_cached_slim_moe_causallm.h"
#include "qwen3_causallm.h"
#include "qwen3_embedding.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"

using json = nlohmann::json;

namespace {

struct ServerConfig {
  std::string model_path;
  std::string host = "127.0.0.1";
  unsigned short port = 8000;
  std::string model_id = "quick.ai";
  bool use_chat_template = true;
  bool verbose = false;
};

struct CompletionResult {
  std::string text;
  TransformerPerformanceMetrics metrics;
};

struct HttpRequest {
  std::string method;
  std::string target;
  std::string path;
  std::map<std::string, std::string> headers;
  std::string body;
};

struct HttpResponse {
  int status = 200;
  std::string reason = "OK";
  std::string content_type = "application/json";
  std::string body;
};

static std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

static std::string trim(const std::string &value) {
  const char *ws = " \t\r\n";
  const size_t first = value.find_first_not_of(ws);
  if (first == std::string::npos)
    return "";
  const size_t last = value.find_last_not_of(ws);
  return value.substr(first, last - first + 1);
}

static std::string now_id(const std::string &prefix) {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto micros =
    std::chrono::duration_cast<std::chrono::microseconds>(now).count();
  std::ostringstream ss;
  ss << prefix << std::hex << micros;
  return ss.str();
}

static long long unix_time_seconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(
           std::chrono::system_clock::now().time_since_epoch())
    .count();
}

static void close_socket(socket_t socket) {
#if defined(_WIN32)
  closesocket(socket);
#else
  close(socket);
#endif
}

static bool send_all(socket_t socket, const std::string &payload) {
  size_t sent = 0;
  while (sent < payload.size()) {
    const char *data = payload.data() + sent;
    const int remaining = static_cast<int>(payload.size() - sent);
#if defined(_WIN32)
    int n = send(socket, data, remaining, 0);
#else
    ssize_t n = send(socket, data, static_cast<size_t>(remaining), 0);
#endif
    if (n <= 0)
      return false;
    sent += static_cast<size_t>(n);
  }
  return true;
}

static std::string status_reason(int status) {
  switch (status) {
  case 200:
    return "OK";
  case 204:
    return "No Content";
  case 400:
    return "Bad Request";
  case 404:
    return "Not Found";
  case 405:
    return "Method Not Allowed";
  case 413:
    return "Payload Too Large";
  case 500:
    return "Internal Server Error";
  default:
    return "OK";
  }
}

static json error_body(const std::string &message,
                       const std::string &type = "invalid_request_error",
                       const std::string &param = "") {
  return {
    {"error",
     {{"message", message},
      {"type", type},
      {"param", param.empty() ? json(nullptr) : json(param)},
      {"code", nullptr}}},
  };
}

static HttpResponse json_response(int status, const json &body) {
  HttpResponse response;
  response.status = status;
  response.reason = status_reason(status);
  response.body = body.dump();
  return response;
}

static HttpResponse sse_response(const std::vector<json> &events) {
  HttpResponse response;
  response.status = 200;
  response.reason = status_reason(200);
  response.content_type = "text/event-stream";

  std::ostringstream body;
  for (const auto &event : events) {
    body << "data: " << event.dump() << "\n\n";
  }
  body << "data: [DONE]\n\n";
  response.body = body.str();
  return response;
}

static std::string serialize_response(const HttpResponse &response) {
  std::ostringstream ss;
  ss << "HTTP/1.1 " << response.status << ' ' << response.reason << "\r\n";
  ss << "Content-Type: " << response.content_type << "\r\n";
  if (response.content_type == "text/event-stream") {
    ss << "Cache-Control: no-cache\r\n";
    ss << "X-Accel-Buffering: no\r\n";
  }
  ss << "Access-Control-Allow-Origin: *\r\n";
  ss << "Access-Control-Allow-Headers: authorization, content-type\r\n";
  ss << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
  ss << "Connection: close\r\n";
  ss << "Content-Length: " << response.body.size() << "\r\n\r\n";
  ss << response.body;
  return ss.str();
}

static bool parse_http_request(std::string raw, HttpRequest &request,
                               std::string &error) {
  const size_t header_end = raw.find("\r\n\r\n");
  if (header_end == std::string::npos) {
    error = "Malformed HTTP request";
    return false;
  }

  std::istringstream headers(raw.substr(0, header_end));
  std::string request_line;
  std::getline(headers, request_line);
  request_line = trim(request_line);

  std::istringstream request_line_stream(request_line);
  std::string version;
  request_line_stream >> request.method >> request.target >> version;
  if (request.method.empty() || request.target.empty()) {
    error = "Malformed HTTP request line";
    return false;
  }

  const size_t query = request.target.find('?');
  request.path = request.target.substr(0, query);

  std::string line;
  while (std::getline(headers, line)) {
    const size_t colon = line.find(':');
    if (colon == std::string::npos)
      continue;
    std::string key = lowercase(trim(line.substr(0, colon)));
    std::string value = trim(line.substr(colon + 1));
    request.headers[key] = value;
  }

  request.body = raw.substr(header_end + 4);
  return true;
}

static bool receive_http_request(socket_t socket, HttpRequest &request,
                                 std::string &error) {
  std::string raw;
  char buffer[4096];
  size_t header_end = std::string::npos;

  while (header_end == std::string::npos) {
#if defined(_WIN32)
    int received = recv(socket, buffer, sizeof(buffer), 0);
#else
    ssize_t received = recv(socket, buffer, sizeof(buffer), 0);
#endif
    if (received <= 0) {
      error = "Failed to read HTTP request";
      return false;
    }
    raw.append(buffer, static_cast<size_t>(received));
    if (raw.size() > 16 * 1024 * 1024) {
      error = "HTTP request is too large";
      return false;
    }
    header_end = raw.find("\r\n\r\n");
  }

  HttpRequest partial;
  if (!parse_http_request(raw, partial, error))
    return false;

  size_t content_length = 0;
  auto it = partial.headers.find("content-length");
  if (it != partial.headers.end()) {
    try {
      content_length = static_cast<size_t>(std::stoul(it->second));
    } catch (...) {
      error = "Invalid Content-Length header";
      return false;
    }
  }

  if (content_length > 16 * 1024 * 1024) {
    error = "HTTP request body is too large";
    return false;
  }

  while (partial.body.size() < content_length) {
#if defined(_WIN32)
    int received = recv(socket, buffer, sizeof(buffer), 0);
#else
    ssize_t received = recv(socket, buffer, sizeof(buffer), 0);
#endif
    if (received <= 0) {
      error = "Unexpected EOF while reading HTTP request body";
      return false;
    }
    partial.body.append(buffer, static_cast<size_t>(received));
  }

  if (partial.body.size() > content_length)
    partial.body.resize(content_length);

  request = std::move(partial);
  return true;
}

static std::string resolve_architecture(std::string model_type,
                                        const std::string &architecture) {
  std::transform(model_type.begin(), model_type.end(), model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (model_type == "embedding") {
    if (architecture == "Qwen3ForCausalLM")
      return "Qwen3Embedding";
    if (architecture == "Gemma3ForCausalLM" ||
        architecture == "Gemma3TextModel")
      return "EmbeddingGemma";
    if (architecture == "Qwen2Model")
      return "Qwen2Embedding";
    throw std::invalid_argument(
      "Unsupported architecture for embedding model: " + architecture);
  }

  return architecture;
}

static void register_models() {
  static std::once_flag once;
  std::call_once(once, []() {
    quick_dot_ai::Factory::Instance().registerModel(
      "LlamaForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<quick_dot_ai::CausalLM>(cfg, generation_cfg,
                                                        nntr_cfg);
      });
    quick_dot_ai::Factory::Instance().registerModel(
      "Qwen2ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<quick_dot_ai::Qwen2CausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
    quick_dot_ai::Factory::Instance().registerModel(
      "Qwen2Embedding", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<quick_dot_ai::Qwen2Embedding>(
          cfg, generation_cfg, nntr_cfg);
      });
    quick_dot_ai::Factory::Instance().registerModel(
      "Qwen3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<quick_dot_ai::Qwen3CausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
    quick_dot_ai::Factory::Instance().registerModel(
      "Qwen3MoeForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<quick_dot_ai::Qwen3MoECausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
    quick_dot_ai::Factory::Instance().registerModel(
      "Qwen3SlimMoeForCausalLM",
      [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<quick_dot_ai::Qwen3SlimMoECausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
    quick_dot_ai::Factory::Instance().registerModel(
      "Qwen3CachedSlimMoeForCausalLM",
      [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<quick_dot_ai::Qwen3CachedSlimMoECausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
    quick_dot_ai::Factory::Instance().registerModel(
      "Qwen3Embedding", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<quick_dot_ai::Qwen3Embedding>(
          cfg, generation_cfg, nntr_cfg);
      });
    quick_dot_ai::Factory::Instance().registerModel(
      "GptOssForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<quick_dot_ai::GptOssForCausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
    quick_dot_ai::Factory::Instance().registerModel(
      "GptOssCachedSlimCausalLM",
      [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<quick_dot_ai::GptOssCachedSlimCausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
    quick_dot_ai::Factory::Instance().registerModel(
      "Gemma3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<quick_dot_ai::Gemma3CausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
    quick_dot_ai::Factory::Instance().registerModel(
      "EmbeddingGemma", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<quick_dot_ai::EmbeddingGemma>(
          cfg, generation_cfg, nntr_cfg);
      });
  });
}

#if defined(_WIN32)
static std::wstring utf8_to_wstring(const std::string &text) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.from_bytes(text);
}
#endif

class OpenAIModel {
public:
  explicit OpenAIModel(ServerConfig config) : config_(std::move(config)) {}

  void load() {
    register_models();

    json cfg = quick_dot_ai::LoadJsonFile(config_.model_path + "/config.json");
    json generation_cfg = quick_dot_ai::LoadJsonFile(config_.model_path +
                                                     "/generation_config.json");
    json nntr_cfg =
      quick_dot_ai::LoadJsonFile(config_.model_path + "/nntr_config.json");

    if (nntr_cfg.contains("tokenizer_file")) {
      std::filesystem::path tokenizer_path =
        nntr_cfg["tokenizer_file"].get<std::string>();
      if (tokenizer_path.is_relative()) {
        tokenizer_path =
          std::filesystem::path(config_.model_path) / tokenizer_path;
        nntr_cfg["tokenizer_file"] = tokenizer_path.string();
      }
    }

    if (nntr_cfg.contains("system_prompt")) {
      system_head_prompt_ =
        nntr_cfg["system_prompt"].value("head_prompt", std::string());
      system_tail_prompt_ =
        nntr_cfg["system_prompt"].value("tail_prompt", std::string());
    }

    std::string architecture =
      cfg["architectures"].get<std::vector<std::string>>()[0];
    if (nntr_cfg.contains("model_type")) {
      architecture = resolve_architecture(
        nntr_cfg["model_type"].get<std::string>(), architecture);
    }

    const std::string tokenizer_config =
      config_.model_path + "/tokenizer_config.json";
    if (std::filesystem::exists(tokenizer_config)) {
      chat_template_ = quick_dot_ai::ChatTemplate::fromFile(tokenizer_config);
      if (chat_template_.isAvailable()) {
        std::cout << "[Info] Chat template loaded from tokenizer_config.json"
                  << std::endl;
      }
    }

    auto model = quick_dot_ai::Factory::Instance().create(
      architecture, cfg, generation_cfg, nntr_cfg);
    if (!model) {
      std::ostringstream os;
      os << "Unknown architecture: " << architecture
         << ". Registered architectures:";
      quick_dot_ai::Factory::Instance().printRegistered(os);
      throw std::runtime_error(os.str());
    }

    causal_lm_ = dynamic_cast<quick_dot_ai::CausalLM *>(model.get());
    if (!causal_lm_) {
      throw std::runtime_error(
        "OpenAI-compatible server currently requires a causal LM model");
    }

    std::string weight_file =
      config_.model_path + "/" + nntr_cfg["model_file_name"].get<std::string>();
    model->initialize();
    model->load_weight(weight_file);
    model_ = std::move(model);
  }

  bool has_chat_template() const { return chat_template_.isAvailable(); }

  std::string render_chat_prompt(
    const std::vector<quick_dot_ai::ChatMessage> &messages) const {
    if (config_.use_chat_template && chat_template_.isAvailable()) {
      std::string rendered = chat_template_.apply(messages);
      if (!rendered.empty())
        return rendered;
    }

    std::ostringstream fallback;
    for (const auto &message : messages) {
      if (message.role == "system")
        fallback << "System: ";
      else if (message.role == "assistant")
        fallback << "Assistant: ";
      else
        fallback << "User: ";
      fallback << message.content << "\n";
    }
    fallback << "Assistant: ";
    return fallback.str();
  }

  CompletionResult complete(const std::string &prompt, bool do_sample) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    causal_lm_->resetGenerationState();

#if defined(_WIN32)
    model_->run(utf8_to_wstring(prompt), do_sample,
                utf8_to_wstring(system_head_prompt_),
                utf8_to_wstring(system_tail_prompt_), config_.verbose);
#else
    model_->run(prompt, do_sample, system_head_prompt_, system_tail_prompt_,
                config_.verbose);
#endif

    CompletionResult result;
    result.text = causal_lm_->getOutput(0);
    result.metrics = causal_lm_->getPerformanceMetrics();
    return result;
  }

private:
  ServerConfig config_;
  std::unique_ptr<quick_dot_ai::Transformer> model_;
  quick_dot_ai::CausalLM *causal_lm_ = nullptr;
  quick_dot_ai::ChatTemplate chat_template_;
  std::string system_head_prompt_;
  std::string system_tail_prompt_;
  std::mutex inference_mutex_;
};

class OpenAIServer {
public:
  OpenAIServer(ServerConfig config, OpenAIModel &model) :
    config_(std::move(config)), model_(model) {}

  int run() {
#if defined(_WIN32)
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
      std::cerr << "Failed to initialize Winsock" << std::endl;
      return EXIT_FAILURE;
    }
#endif

    socket_t listen_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_socket == invalid_socket_value) {
      std::cerr << "Failed to create server socket" << std::endl;
      cleanup_network();
      return EXIT_FAILURE;
    }

    int reuse = 1;
    setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char *>(&reuse), sizeof(reuse));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(config_.port);
    if (inet_pton(AF_INET, config_.host.c_str(), &address.sin_addr) != 1) {
      std::cerr << "Invalid IPv4 host: " << config_.host << std::endl;
      close_socket(listen_socket);
      cleanup_network();
      return EXIT_FAILURE;
    }

    if (bind(listen_socket, reinterpret_cast<sockaddr *>(&address),
             sizeof(address)) < 0) {
      std::cerr << "Failed to bind " << config_.host << ':' << config_.port
                << std::endl;
      close_socket(listen_socket);
      cleanup_network();
      return EXIT_FAILURE;
    }

    if (listen(listen_socket, 32) < 0) {
      std::cerr << "Failed to listen on " << config_.host << ':' << config_.port
                << std::endl;
      close_socket(listen_socket);
      cleanup_network();
      return EXIT_FAILURE;
    }

    std::cout << "Quick.AI OpenAI-compatible server listening on http://"
              << config_.host << ':' << config_.port << std::endl;
    std::cout << "Model id: " << config_.model_id << std::endl;

    while (true) {
      sockaddr_in client_address{};
#if defined(_WIN32)
      int client_len = sizeof(client_address);
#else
      socklen_t client_len = sizeof(client_address);
#endif
      socket_t client =
        accept(listen_socket, reinterpret_cast<sockaddr *>(&client_address),
               &client_len);
      if (client == invalid_socket_value)
        continue;

      std::thread([this, client]() {
        handle_client(client);
        close_socket(client);
      }).detach();
    }

    close_socket(listen_socket);
    cleanup_network();
    return EXIT_SUCCESS;
  }

private:
  static void cleanup_network() {
#if defined(_WIN32)
    WSACleanup();
#endif
  }

  void handle_client(socket_t client) {
    HttpRequest request;
    std::string error;
    if (!receive_http_request(client, request, error)) {
      HttpResponse response = json_response(400, error_body(error));
      send_all(client, serialize_response(response));
      return;
    }

    HttpResponse response = route(request);
    send_all(client, serialize_response(response));
  }

  HttpResponse route(const HttpRequest &request) {
    try {
      if (request.method == "OPTIONS") {
        HttpResponse response;
        response.status = 204;
        response.reason = status_reason(204);
        return response;
      }

      if (request.method == "GET" &&
          (request.path == "/" || request.path == "/health")) {
        return json_response(200,
                             {{"status", "ok"},
                              {"object", "quick.ai.server"},
                              {"model", config_.model_id},
                              {"chat_template", model_.has_chat_template()}});
      }

      if (request.method == "GET" && request.path == "/v1/models")
        return handle_models();

      if (request.method == "POST" && request.path == "/v1/completions")
        return handle_completions(request);

      if (request.method == "POST" && request.path == "/v1/chat/completions")
        return handle_chat_completions(request);

      if (request.path == "/v1/completions" ||
          request.path == "/v1/chat/completions") {
        return json_response(405, error_body("Method not allowed"));
      }

      return json_response(404, error_body("Endpoint not found"));
    } catch (const json::parse_error &e) {
      return json_response(
        400, error_body(std::string("Invalid JSON: ") + e.what()));
    } catch (const json::type_error &e) {
      return json_response(
        400, error_body(std::string("Invalid JSON type: ") + e.what()));
    } catch (const std::invalid_argument &e) {
      return json_response(400, error_body(e.what()));
    } catch (const std::exception &e) {
      return json_response(500, error_body(e.what(), "server_error"));
    }
  }

  HttpResponse handle_models() const {
    json body = {
      {"object", "list"},
      {"data", json::array({{{"id", config_.model_id},
                             {"object", "model"},
                             {"created", unix_time_seconds()},
                             {"owned_by", "quick.ai"}}})},
    };
    return json_response(200, body);
  }

  HttpResponse handle_completions(const HttpRequest &request) {
    json body = json::parse(request.body.empty() ? "{}" : request.body);
    const bool stream = quick_dot_ai::server::json_bool(body, "stream", false);

    std::vector<std::string> prompts =
      quick_dot_ai::server::parse_completion_prompts(body);
    const bool do_sample = quick_dot_ai::server::sampling_requested(body);
    json choices = json::array();
    std::vector<json> stream_events;
    const std::string id = now_id("cmpl-");
    const long long created = unix_time_seconds();
    const std::string model = body.value("model", config_.model_id);
    unsigned int prompt_tokens = 0;
    unsigned int completion_tokens = 0;

    for (size_t i = 0; i < prompts.size(); ++i) {
      CompletionResult result = model_.complete(prompts[i], do_sample);
      if (stream) {
        stream_events.push_back(
          {{"id", id},
           {"object", "text_completion"},
           {"created", created},
           {"model", model},
           {"choices", json::array({{{"text", result.text},
                                     {"index", static_cast<int>(i)},
                                     {"logprobs", nullptr},
                                     {"finish_reason", nullptr}}})}});
        stream_events.push_back(
          {{"id", id},
           {"object", "text_completion"},
           {"created", created},
           {"model", model},
           {"choices", json::array({{{"text", ""},
                                     {"index", static_cast<int>(i)},
                                     {"logprobs", nullptr},
                                     {"finish_reason", "stop"}}})}});
      }
      choices.push_back({{"text", result.text},
                         {"index", static_cast<int>(i)},
                         {"logprobs", nullptr},
                         {"finish_reason", "stop"}});
      prompt_tokens += result.metrics.prefill_tokens;
      completion_tokens += result.metrics.generation_tokens;
    }

    if (stream)
      return sse_response(stream_events);

    json response = {
      {"id", id},
      {"object", "text_completion"},
      {"created", created},
      {"model", model},
      {"choices", choices},
      {"usage",
       {{"prompt_tokens", prompt_tokens},
        {"completion_tokens", completion_tokens},
        {"total_tokens", prompt_tokens + completion_tokens}}},
    };
    return json_response(200, response);
  }

  HttpResponse handle_chat_completions(const HttpRequest &request) {
    json body = json::parse(request.body.empty() ? "{}" : request.body);
    const bool stream = quick_dot_ai::server::json_bool(body, "stream", false);

    std::vector<quick_dot_ai::ChatMessage> messages =
      quick_dot_ai::server::parse_chat_messages(body);
    const std::string prompt = model_.render_chat_prompt(messages);
    CompletionResult result =
      model_.complete(prompt, quick_dot_ai::server::sampling_requested(body));
    const std::string id = now_id("chatcmpl-");
    const long long created = unix_time_seconds();
    const std::string model = body.value("model", config_.model_id);

    if (stream) {
      return sse_response({
        {{"id", id},
         {"object", "chat.completion.chunk"},
         {"created", created},
         {"model", model},
         {"choices", json::array({{{"index", 0},
                                   {"delta", {{"role", "assistant"}}},
                                   {"finish_reason", nullptr}}})}},
        {{"id", id},
         {"object", "chat.completion.chunk"},
         {"created", created},
         {"model", model},
         {"choices", json::array({{{"index", 0},
                                   {"delta", {{"content", result.text}}},
                                   {"finish_reason", nullptr}}})}},
        {{"id", id},
         {"object", "chat.completion.chunk"},
         {"created", created},
         {"model", model},
         {"choices", json::array({{{"index", 0},
                                   {"delta", json::object()},
                                   {"finish_reason", "stop"}}})}},
      });
    }

    json response = {
      {"id", id},
      {"object", "chat.completion"},
      {"created", created},
      {"model", model},
      {"choices",
       json::array(
         {{{"index", 0},
           {"message", {{"role", "assistant"}, {"content", result.text}}},
           {"finish_reason", "stop"}}})},
      {"usage", quick_dot_ai::server::usage_json(result.metrics)},
    };
    return json_response(200, response);
  }

  ServerConfig config_;
  OpenAIModel &model_;
};

static void print_usage(const char *argv0) {
  std::cout << "Usage: " << argv0
            << " <model_path> [--host 127.0.0.1] [--port 8000]\n"
               "                 [--model-id quick.ai] [--no-chat-template]\n"
               "                 [--verbose]\n\n"
               "Serves OpenAI-compatible REST endpoints:\n"
               "  GET  /v1/models\n"
               "  POST /v1/completions\n"
               "  POST /v1/chat/completions\n";
}

static ServerConfig parse_args(int argc, char **argv) {
  if (argc >= 2) {
    std::string first = argv[1];
    if (first == "--help" || first == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
  }

  if (argc < 2) {
    print_usage(argv[0]);
    throw std::invalid_argument("missing model_path");
  }

  ServerConfig config;
  config.model_path = argv[1];

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    auto require_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc)
        throw std::invalid_argument(name + " requires a value");
      return argv[++i];
    };

    if (arg == "--host") {
      config.host = require_value(arg);
    } else if (arg == "--port") {
      const int port = std::stoi(require_value(arg));
      if (port <= 0 || port > 65535)
        throw std::invalid_argument("--port must be between 1 and 65535");
      config.port = static_cast<unsigned short>(port);
    } else if (arg == "--model-id") {
      config.model_id = require_value(arg);
    } else if (arg == "--no-chat-template") {
      config.use_chat_template = false;
    } else if (arg == "--verbose") {
      config.verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }

  return config;
}

} // namespace

int main(int argc, char **argv) {
  try {
    ServerConfig config = parse_args(argc, argv);
    OpenAIModel model(config);
    model.load();

    OpenAIServer server(config, model);
    return server.run();
  } catch (const std::exception &e) {
    std::cerr << "[!] " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}
