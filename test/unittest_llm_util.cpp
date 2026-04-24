// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_llm_util.cpp
 * @date   24 Apr 2026
 * @brief  GoogleTest unit tests for the logits post-processing helpers
 *         declared in llm_util.hpp. These tests do not depend on any model
 *         weights - they exercise pure numerical routines.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <llm_util.hpp>

namespace {

TEST(LlmUtilRepetitionPenalty, NegativeLogitIsMultiplied) {
  std::vector<float> logits = {-2.0f, 1.0f, -0.5f, 4.0f};
  std::vector<unsigned int> input_ids = {0, 2};
  const float penalty = 2.0f;

  applyRepetitionPenalty(logits.data(), input_ids.data(),
                         static_cast<unsigned int>(input_ids.size()), penalty);

  EXPECT_FLOAT_EQ(logits[0], -4.0f); // negative: multiplied by penalty
  EXPECT_FLOAT_EQ(logits[1], 1.0f);  // untouched
  EXPECT_FLOAT_EQ(logits[2], -1.0f); // negative: multiplied by penalty
  EXPECT_FLOAT_EQ(logits[3], 4.0f);  // untouched
}

TEST(LlmUtilRepetitionPenalty, PositiveLogitIsDivided) {
  std::vector<float> logits = {2.0f, 3.0f, 4.0f};
  std::vector<unsigned int> input_ids = {0, 1, 2};
  const float penalty = 2.0f;

  applyRepetitionPenalty(logits.data(), input_ids.data(),
                         static_cast<unsigned int>(input_ids.size()), penalty);

  EXPECT_FLOAT_EQ(logits[0], 1.0f);
  EXPECT_FLOAT_EQ(logits[1], 1.5f);
  EXPECT_FLOAT_EQ(logits[2], 2.0f);
}

TEST(LlmUtilBadWordsPenalty, ForcesNegativeInfinity) {
  std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<unsigned int> bad_ids = {1, 3};

  applyBadWordsPenalty(logits.data(), bad_ids.data(),
                       static_cast<unsigned int>(bad_ids.size()));

  EXPECT_FLOAT_EQ(logits[0], 1.0f);
  EXPECT_TRUE(std::isinf(logits[1]) && logits[1] < 0);
  EXPECT_FLOAT_EQ(logits[2], 3.0f);
  EXPECT_TRUE(std::isinf(logits[3]) && logits[3] < 0);
  EXPECT_FLOAT_EQ(logits[4], 5.0f);
}

TEST(LlmUtilGenerateMultiTokens, ReturnsTopKIndicesByLogit) {
  std::vector<float> logits = {0.1f, 2.5f, -1.0f, 4.2f, 3.3f, 0.0f};

  auto top = generate_multi_tokens(
    logits.data(), static_cast<unsigned int>(logits.size()),
    /*NUM_TARGET_TOKENS=*/3, /*repetition_penalty=*/1.0f,
    /*input_ids=*/nullptr, /*NUM_INPUT_IDS=*/0,
    /*bad_words_ids=*/nullptr, /*NUM_BAD_WORDS_IDS=*/0);

  ASSERT_EQ(top.size(), 3u);
  EXPECT_EQ(top[0], 3u); // 4.2
  EXPECT_EQ(top[1], 4u); // 3.3
  EXPECT_EQ(top[2], 1u); // 2.5
}

TEST(LlmUtilGenerateMultiTokens, BadWordsAreNeverSelected) {
  std::vector<float> logits = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  std::vector<unsigned int> bad_ids = {0}; // best-scoring token is forbidden

  auto top = generate_multi_tokens(
    logits.data(), static_cast<unsigned int>(logits.size()),
    /*NUM_TARGET_TOKENS=*/1, /*repetition_penalty=*/1.0f,
    /*input_ids=*/nullptr, /*NUM_INPUT_IDS=*/0, bad_ids.data(),
    static_cast<unsigned int>(bad_ids.size()));

  ASSERT_EQ(top.size(), 1u);
  EXPECT_EQ(top[0], 1u); // the formerly-second-best token wins
}

TEST(LlmUtilGenerateMultiTokens, RepetitionPenaltyDemotesRepeats) {
  // Token 0 scores highest initially. After repetition penalty it must
  // fall below token 1.
  std::vector<float> logits = {4.0f, 3.0f, 2.0f};
  std::vector<unsigned int> input_ids = {0};
  const float penalty = 2.0f;

  auto top = generate_multi_tokens(
    logits.data(), static_cast<unsigned int>(logits.size()),
    /*NUM_TARGET_TOKENS=*/1, penalty, input_ids.data(),
    static_cast<unsigned int>(input_ids.size()),
    /*bad_words_ids=*/nullptr, /*NUM_BAD_WORDS_IDS=*/0);

  ASSERT_EQ(top.size(), 1u);
  EXPECT_EQ(top[0], 1u) << "token 0 should have been demoted by the penalty";
}

TEST(LlmUtilApplyTKP, DividesByTemperatureAndReturnsMax) {
  std::vector<float> logits = {2.0f, 4.0f, 1.0f};

  float max_logit = applyTKP(logits.data(), static_cast<int>(logits.size()),
                             /*temperature=*/2.0f, /*top_k=*/0, /*top_p=*/0.0f);

  EXPECT_FLOAT_EQ(logits[0], 1.0f);
  EXPECT_FLOAT_EQ(logits[1], 2.0f);
  EXPECT_FLOAT_EQ(logits[2], 0.5f);
  EXPECT_FLOAT_EQ(max_logit, 2.0f);
}

TEST(LlmUtilApplyTKP, SkipsTemperatureScalingWhenBelowThreshold) {
  // A temperature of zero triggers the 1e-5 guard in applyTKP and the
  // routine must leave logits untouched.
  std::vector<float> logits = {1.0f, 2.0f, 3.0f};

  float max_logit = applyTKP(logits.data(), static_cast<int>(logits.size()),
                             /*temperature=*/0.0f, /*top_k=*/0, /*top_p=*/0.0f);

  EXPECT_FLOAT_EQ(logits[0], 1.0f);
  EXPECT_FLOAT_EQ(logits[1], 2.0f);
  EXPECT_FLOAT_EQ(logits[2], 3.0f);
  EXPECT_FLOAT_EQ(max_logit, 3.0f);
}

} // namespace
