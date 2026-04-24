// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_model_config.cpp
 * @date   24 Apr 2026
 * @brief  GoogleTest unit tests for the model architecture / runtime
 *         registration API exposed through model_config_internal.h. These
 *         tests do not require any model binary: they only exercise the
 *         registry bookkeeping and the validation of null parameters.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <climits>
#include <cstring>

#include "causal_lm_api.h"
#include "model_config_internal.h"

namespace {

ModelArchConfig make_dummy_arch(const char *name) {
  ModelArchConfig ac;
  std::memset(&ac, 0, sizeof(ac));
  ac.vocab_size = 32000;
  ac.hidden_size = 64;
  ac.intermediate_size = 128;
  ac.num_hidden_layers = 2;
  ac.num_attention_heads = 4;
  ac.head_dim = 16;
  ac.num_key_value_heads = 2;
  ac.max_position_embeddings = 1024;
  ac.rope_theta = 10000.0f;
  ac.rms_norm_eps = 1e-6f;
  ac.tie_word_embeddings = false;
  ac.sliding_window = UINT_MAX;
  ac.sliding_window_pattern = 0;
  std::strncpy(ac.architecture, name, sizeof(ac.architecture) - 1);
  ac.bos_token_id = 1;
  ac.eos_token_ids[0] = 2;
  ac.num_eos_token_ids = 1;
  return ac;
}

ModelRuntimeConfig make_dummy_runtime() {
  ModelRuntimeConfig rc;
  std::memset(&rc, 0, sizeof(rc));
  rc.batch_size = 1;
  std::strncpy(rc.model_type, "CausalLM", sizeof(rc.model_type) - 1);
  std::strncpy(rc.model_tensor_type, "FP32-FP32",
               sizeof(rc.model_tensor_type) - 1);
  rc.init_seq_len = 64;
  rc.max_seq_len = 128;
  rc.num_to_generate = 16;
  rc.fsu = false;
  rc.fsu_lookahead = 0;
  std::strncpy(rc.embedding_dtype, "FP32", sizeof(rc.embedding_dtype) - 1);
  std::strncpy(rc.fc_layer_dtype, "FP32", sizeof(rc.fc_layer_dtype) - 1);
  std::strncpy(rc.lmhead_dtype, "FP32", sizeof(rc.lmhead_dtype) - 1);
  std::strncpy(rc.model_file_name, "dummy.bin",
               sizeof(rc.model_file_name) - 1);
  std::strncpy(rc.tokenizer_file, "tokenizer.json",
               sizeof(rc.tokenizer_file) - 1);
  rc.num_bad_word_ids = 0;
  rc.top_k = 20;
  rc.top_p = 0.95f;
  rc.temperature = 0.7f;
  return rc;
}

TEST(ModelConfigApi, RegisterArchitectureRejectsNullName) {
  ModelArchConfig ac = make_dummy_arch("UT-Arch");
  EXPECT_EQ(registerModelArchitecture(nullptr, ac),
            CAUSAL_LM_ERROR_INVALID_PARAMETER);
}

TEST(ModelConfigApi, RegisterModelRejectsNullName) {
  ModelRuntimeConfig rc = make_dummy_runtime();
  EXPECT_EQ(registerModel(nullptr, "UT-Arch", rc),
            CAUSAL_LM_ERROR_INVALID_PARAMETER);
  EXPECT_EQ(registerModel("UT-Model", nullptr, rc),
            CAUSAL_LM_ERROR_INVALID_PARAMETER);
}

TEST(ModelConfigApi, RegisterArchitectureAndModelSucceed) {
  ModelArchConfig ac = make_dummy_arch("UT-Arch");
  EXPECT_EQ(registerModelArchitecture("UT-Arch-Unique", ac),
            CAUSAL_LM_ERROR_NONE);

  ModelRuntimeConfig rc = make_dummy_runtime();
  EXPECT_EQ(registerModel("UT-Model-Unique", "UT-Arch-Unique", rc),
            CAUSAL_LM_ERROR_NONE);
}

TEST(ModelConfigApi, BuiltinQwen3RegistersWithoutError) {
  // Idempotent: may have been invoked already by other tests that touched
  // the C API. Re-running must still succeed.
  EXPECT_EQ(register_builtin_model_configs(), 0);
}

TEST(ModelConfigApi, SetOptionsAcceptsAllFlagCombinations) {
  Config cfg;
  cfg.use_chat_template = true;
  cfg.debug_mode = false;
  cfg.verbose = false;
  EXPECT_EQ(setOptions(cfg), CAUSAL_LM_ERROR_NONE);

  cfg.use_chat_template = false;
  cfg.debug_mode = false;
  cfg.verbose = true;
  EXPECT_EQ(setOptions(cfg), CAUSAL_LM_ERROR_NONE);
}

} // namespace
