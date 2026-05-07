// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    openai_protocol.cpp
 * @brief   Pure OpenAI-compatible request/response helpers.
 */

#include "openai_protocol.h"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace quick_dot_ai {
namespace server {

bool json_bool(const json &body, const std::string &key, bool default_value) {
  if (!body.contains(key))
    return default_value;
  if (body[key].is_boolean())
    return body[key].get<bool>();
  return default_value;
}

bool sampling_requested(const json &body) {
  if (json_bool(body, "do_sample", false))
    return true;
  if (body.contains("temperature") && body["temperature"].is_number()) {
    return body["temperature"].get<double>() > 0.0;
  }
  return false;
}

std::string message_content_to_text(const json &content) {
  if (content.is_string())
    return content.get<std::string>();

  if (content.is_array()) {
    std::ostringstream ss;
    bool has_text = false;
    for (const auto &part : content) {
      if (part.is_object() && part.value("type", "") == "text" &&
          part.contains("text")) {
        if (has_text)
          ss << "\n";
        ss << part["text"].get<std::string>();
        has_text = true;
      }
    }
    return ss.str();
  }

  if (content.is_null())
    return "";
  return content.dump();
}

std::vector<quick_dot_ai::ChatMessage> parse_chat_messages(const json &body) {
  if (!body.contains("messages") || !body["messages"].is_array()) {
    throw std::invalid_argument("`messages` must be an array");
  }

  std::vector<quick_dot_ai::ChatMessage> messages;
  for (const auto &entry : body["messages"]) {
    if (!entry.is_object())
      throw std::invalid_argument("Each message must be an object");

    quick_dot_ai::ChatMessage message;
    message.role = entry.value("role", "user");
    message.content =
      message_content_to_text(entry.value("content", json(nullptr)));
    messages.push_back(std::move(message));
  }

  if (messages.empty())
    throw std::invalid_argument("`messages` must not be empty");
  return messages;
}

std::vector<std::string> parse_completion_prompts(const json &body) {
  if (!body.contains("prompt"))
    throw std::invalid_argument("`prompt` is required");

  const json &prompt = body["prompt"];
  if (prompt.is_string())
    return {prompt.get<std::string>()};

  if (prompt.is_array()) {
    std::vector<std::string> prompts;
    for (const auto &entry : prompt) {
      if (!entry.is_string()) {
        throw std::invalid_argument(
          "Only string prompts are supported in prompt arrays");
      }
      prompts.push_back(entry.get<std::string>());
    }
    if (prompts.empty())
      throw std::invalid_argument("`prompt` array must not be empty");
    return prompts;
  }

  throw std::invalid_argument("`prompt` must be a string or string array");
}

json usage_json(const TransformerPerformanceMetrics &metrics) {
  const unsigned int total = metrics.prefill_tokens + metrics.generation_tokens;
  return {{"prompt_tokens", metrics.prefill_tokens},
          {"completion_tokens", metrics.generation_tokens},
          {"total_tokens", total}};
}

} // namespace server
} // namespace quick_dot_ai
