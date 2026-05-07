// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_openai_protocol.cpp
 * @brief  GoogleTest unit tests for OpenAI-compatible request helpers.
 */

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "server/openai_protocol.h"

namespace {

using quick_dot_ai::server::json;

TEST(OpenAIProtocol, ParsesStringPrompt) {
  json body = {{"prompt", "Quick.AI is"}};

  auto prompts = quick_dot_ai::server::parse_completion_prompts(body);

  ASSERT_EQ(prompts.size(), 1u);
  EXPECT_EQ(prompts[0], "Quick.AI is");
}

TEST(OpenAIProtocol, ParsesPromptArray) {
  json body = {{"prompt", json::array({"first", "second"})}};

  auto prompts = quick_dot_ai::server::parse_completion_prompts(body);

  ASSERT_EQ(prompts.size(), 2u);
  EXPECT_EQ(prompts[0], "first");
  EXPECT_EQ(prompts[1], "second");
}

TEST(OpenAIProtocol, RejectsMissingPrompt) {
  EXPECT_THROW(quick_dot_ai::server::parse_completion_prompts(json::object()),
               std::invalid_argument);
}

TEST(OpenAIProtocol, RejectsNonStringPromptArrayEntries) {
  json body = {{"prompt", json::array({"valid", 3})}};

  EXPECT_THROW(quick_dot_ai::server::parse_completion_prompts(body),
               std::invalid_argument);
}

TEST(OpenAIProtocol, ParsesChatMessagesWithStringContent) {
  json body = {
    {"messages", json::array({{{"role", "system"}, {"content", "brief"}},
                              {{"role", "user"}, {"content", "hello"}}})}};

  auto messages = quick_dot_ai::server::parse_chat_messages(body);

  ASSERT_EQ(messages.size(), 2u);
  EXPECT_EQ(messages[0].role, "system");
  EXPECT_EQ(messages[0].content, "brief");
  EXPECT_EQ(messages[1].role, "user");
  EXPECT_EQ(messages[1].content, "hello");
}

TEST(OpenAIProtocol, ParsesChatContentParts) {
  json content = json::array({
    {{"type", "text"}, {"text", "line one"}},
    {{"type", "image_url"}, {"image_url", {{"url", "ignored"}}}},
    {{"type", "text"}, {"text", "line two"}},
  });
  json body = {
    {"messages", json::array({{{"role", "user"}, {"content", content}}})}};

  auto messages = quick_dot_ai::server::parse_chat_messages(body);

  ASSERT_EQ(messages.size(), 1u);
  EXPECT_EQ(messages[0].content, "line one\nline two");
}

TEST(OpenAIProtocol, RejectsEmptyMessages) {
  json body = {{"messages", json::array()}};

  EXPECT_THROW(quick_dot_ai::server::parse_chat_messages(body),
               std::invalid_argument);
}

TEST(OpenAIProtocol, SamplingRequestedByDoSampleOrTemperature) {
  EXPECT_TRUE(quick_dot_ai::server::sampling_requested({{"do_sample", true}}));
  EXPECT_TRUE(quick_dot_ai::server::sampling_requested({{"temperature", 0.7}}));
  EXPECT_FALSE(
    quick_dot_ai::server::sampling_requested({{"temperature", 0.0}}));
  EXPECT_FALSE(quick_dot_ai::server::sampling_requested(json::object()));
}

TEST(OpenAIProtocol, UsageJsonAddsTotals) {
  TransformerPerformanceMetrics metrics{};
  metrics.prefill_tokens = 7;
  metrics.generation_tokens = 11;

  json usage = quick_dot_ai::server::usage_json(metrics);

  EXPECT_EQ(usage["prompt_tokens"], 7);
  EXPECT_EQ(usage["completion_tokens"], 11);
  EXPECT_EQ(usage["total_tokens"], 18);
}

} // namespace
