// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_tokenizer.cpp
 * @date   24 Apr 2026
 * @brief  GoogleTest unit tests for the HuggingFace tokenizer wrapper used
 *         by Quick.AI. The tests require the Qwen3-0.6B tokenizer.json to
 *         be available on disk; when the file cannot be located the tests
 *         skip gracefully rather than fail. This keeps the unit test suite
 *         usable on machines where the model bundle has not been fetched.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>

#include <tokenizers_cpp.h>

namespace {

bool file_exists(const std::string &path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

std::string slurp(const std::string &path) {
  std::ifstream in(path);
  std::stringstream buf;
  buf << in.rdbuf();
  return buf.str();
}

std::string locate_tokenizer_json() {
  // Highest priority: explicit override from the test harness.
  if (const char *env = std::getenv("QUICKAI_TEST_TOKENIZER_JSON")) {
    if (*env && file_exists(env))
      return env;
  }
  // Directory installed by test/scripts/download_qwen3_0.6b.sh.
  const char *candidates[] = {
    "./models/qwen3-0.6b-w16a16/tokenizer.json",
    "./models/qwen3-0.6b-w4a32/tokenizer.json",
    "./.test_cache/nntrainer-causallm-models/qwen3-0.6b-q40-x86/tokenizer.json",
  };
  for (auto *p : candidates) {
    if (file_exists(p))
      return p;
  }
  return "";
}

class TokenizerFixture : public ::testing::Test {
protected:
  void SetUp() override {
    path_ = locate_tokenizer_json();
    if (path_.empty()) {
      GTEST_SKIP() << "tokenizer.json not found; run "
                      "test/scripts/download_qwen3_0.6b.sh first";
    }
    blob_ = slurp(path_);
    ASSERT_FALSE(blob_.empty()) << "tokenizer.json at " << path_ << " is empty";
    tok_ = tokenizers::Tokenizer::FromBlobJSON(blob_);
    ASSERT_NE(tok_, nullptr);
  }

  std::string path_;
  std::string blob_;
  std::unique_ptr<tokenizers::Tokenizer> tok_;
};

TEST_F(TokenizerFixture, VocabIsNonEmpty) {
  EXPECT_GT(tok_->GetVocabSize(), 0u);
}

TEST_F(TokenizerFixture, EncodeDecodeRoundtrip) {
  const std::string text = "Hello, Quick.AI!";
  auto ids = tok_->Encode(text);
  ASSERT_FALSE(ids.empty());
  std::string decoded = tok_->Decode(ids);
  // Whitespace normalisation varies across tokenizers, so compare after
  // stripping leading/trailing spaces.
  auto trim = [](std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
      s.erase(s.begin());
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
      s.pop_back();
    return s;
  };
  EXPECT_EQ(trim(decoded), text);
}

TEST_F(TokenizerFixture, IdToTokenAndBack) {
  auto ids = tok_->Encode("tokenizer");
  ASSERT_FALSE(ids.empty());
  for (int32_t id : ids) {
    std::string piece = tok_->IdToToken(id);
    ASSERT_FALSE(piece.empty()) << "id " << id << " mapped to empty string";
    int32_t round = tok_->TokenToId(piece);
    EXPECT_EQ(round, id)
      << "round-trip failed for id " << id << " piece=" << piece;
  }
}

TEST_F(TokenizerFixture, EncodeBatchMatchesEncode) {
  const std::vector<std::string> texts = {"foo", "bar baz", "Quick.AI rocks"};
  auto batch = tok_->EncodeBatch(texts);
  ASSERT_EQ(batch.size(), texts.size());
  for (size_t i = 0; i < texts.size(); ++i) {
    EXPECT_EQ(batch[i], tok_->Encode(texts[i])) << "mismatch at index " << i;
  }
}

} // namespace
