// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    openai_protocol.h
 * @brief   Pure OpenAI-compatible request/response helpers.
 */

#ifndef __QUICK_AI_OPENAI_PROTOCOL_H__
#define __QUICK_AI_OPENAI_PROTOCOL_H__

#include <string>
#include <vector>

#include "chat_template.h"
#include "json.hpp"
#include "performance_metrics.h"

namespace quick_dot_ai {
namespace server {

using json = nlohmann::json;

bool json_bool(const json &body, const std::string &key,
               bool default_value = false);

bool sampling_requested(const json &body);

std::string message_content_to_text(const json &content);

std::vector<quick_dot_ai::ChatMessage> parse_chat_messages(const json &body);

std::vector<std::string> parse_completion_prompts(const json &body);

json usage_json(const TransformerPerformanceMetrics &metrics);

} // namespace server
} // namespace quick_dot_ai

#endif // __QUICK_AI_OPENAI_PROTOCOL_H__
