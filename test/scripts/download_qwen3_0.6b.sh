#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# @file   download_qwen3_0.6b.sh
# @date   24 Apr 2026
# @brief  Fetch / build the Qwen3-0.6B (fc=Q4_0, embed=Q6_K, lmhead=Q6_K)
#         bundle from github.com/nntrainer/quick.ai-models and
#         stage it under ./models/qwen3-0.6b-w16a16/ where the Quick.AI C
#         API expects to find it.
# @see    https://github.com/nntrainer/quick.ai-models
# @author Eunju Yang <ej.yang@samsung.com>
#
# The Q6_K embedding / LM head is deliberate. Qwen3-0.6B has
# tie_word_embeddings=true, and the nntrainer tied-embedding layer only
# accepts Q6_K or FP32 weights on the tied path; Q4_0 throws
#   "Tieword embedding is not supported yet for the data type"
# at the first decode step.
#
# Usage:
#   test/scripts/download_qwen3_0.6b.sh [install_root]
#
# Environment:
#   QUICKAI_MODELS_REPO_BRANCH   models repo branch (default: main)
#                                point at a feature branch when iterating on
#                                a not-yet-merged bundle revision
#   QUICK_DOT_AI_QUANTIZE        quantizer binary for local rebuild
#                                (defaults to <install_root>/build/quick_dot_ai_quantize)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INSTALL_ROOT="${1:-$REPO_ROOT}"

MODEL_REPO_URL="https://github.com/nntrainer/quick.ai-models.git"
MODEL_REPO_BRANCH="${QUICKAI_MODELS_REPO_BRANCH:-main}"
MODEL_DIR_NAME="qwen3-0.6b-q40-q6k-x86"
TARGET_DIR_NAME="qwen3-0.6b-w16a16"
BIN_FILE_NAME="nntr_qwen3_0.6b_w4e6a32.bin"

CACHE_DIR="$INSTALL_ROOT/.test_cache/quick.ai-models"
TARGET_DIR="$INSTALL_ROOT/models/$TARGET_DIR_NAME"

log() { echo "[download_qwen3_0.6b] $*"; }

ensure_clone() {
  if [ -d "$CACHE_DIR/.git" ]; then
    log "model cache exists at $CACHE_DIR; fetching latest $MODEL_REPO_BRANCH"
    git -C "$CACHE_DIR" fetch --depth 1 origin "$MODEL_REPO_BRANCH"
    git -C "$CACHE_DIR" checkout -q -B "$MODEL_REPO_BRANCH" "origin/$MODEL_REPO_BRANCH"
    git -C "$CACHE_DIR" reset --hard "origin/$MODEL_REPO_BRANCH"
  else
    log "cloning $MODEL_REPO_URL -> $CACHE_DIR (branch=$MODEL_REPO_BRANCH)"
    mkdir -p "$(dirname "$CACHE_DIR")"
    git clone --depth 1 --branch "$MODEL_REPO_BRANCH" "$MODEL_REPO_URL" "$CACHE_DIR"
  fi
}

has_bin_parts() {
  local src="$CACHE_DIR/$MODEL_DIR_NAME"
  compgen -G "$src/${BIN_FILE_NAME}.part_??" > /dev/null
}

rebuild_bin_from_hf() {
  local recipe="$CACHE_DIR/scripts/convert_qwen3_0.6b_q6k_lmhead.sh"
  if [ ! -f "$recipe" ]; then
    log "ERROR: convert recipe not found at $recipe"
    log "       the model repo is either on an older revision or missing the recipe"
    exit 1
  fi
  chmod +x "$recipe"

  local quantizer="${QUICK_DOT_AI_QUANTIZE:-$INSTALL_ROOT/build/quick_dot_ai_quantize}"
  if [ ! -x "$quantizer" ]; then
    log "ERROR: quantizer not executable: $quantizer"
    log "       build Quick.AI first (-Denable-test=true is fine) or set QUICK_DOT_AI_QUANTIZE"
    exit 1
  fi

  log "bin parts absent, running local rebuild recipe (HF download + quantize)"
  log "  recipe    : $recipe"
  log "  quantizer : $quantizer"
  QUICK_DOT_AI_QUANTIZE="$quantizer" \
  WEIGHT_CONVERTER="$INSTALL_ROOT/subprojects/nntrainer/Applications/CausalLM/res/qwen3/qwen3-4b/weight_converter.py" \
    "$recipe"
}

combine_bin() {
  local src="$CACHE_DIR/$MODEL_DIR_NAME"
  if [ ! -f "$src/combine.sh" ]; then
    log "ERROR: combine.sh not found under $src"
    exit 1
  fi
  log "running combine.sh to reassemble $BIN_FILE_NAME"
  chmod +x "$src/combine.sh"
  ( cd "$src" && ./combine.sh )
}

verify_sha256() {
  local src="$CACHE_DIR/$MODEL_DIR_NAME"
  if [ -f "$src/SHA256SUMS" ]; then
    log "verifying SHA256SUMS for config + tokenizer + part assets"
    ( cd "$src" && sha256sum -c --ignore-missing SHA256SUMS )
  else
    log "SHA256SUMS absent, skipping verification"
  fi
}

install_to_target() {
  local src="$CACHE_DIR/$MODEL_DIR_NAME"
  log "installing into $TARGET_DIR"
  rm -rf "$TARGET_DIR"
  mkdir -p "$TARGET_DIR"

  # Copy JSON / text configs and the reassembled bin. Skip the split parts
  # so the install directory stays lean.
  for f in config.json generation_config.json nntr_config.json \
           tokenizer.json tokenizer_config.json vocab.json merges.txt \
           "$BIN_FILE_NAME"; do
    if [ -f "$src/$f" ]; then
      cp "$src/$f" "$TARGET_DIR/"
    fi
  done
}

main() {
  ensure_clone
  if ! has_bin_parts; then
    rebuild_bin_from_hf
  fi
  combine_bin
  verify_sha256
  install_to_target
  log "done. QUICKAI_TEST_MODEL_DIR=$TARGET_DIR"
  echo "$TARGET_DIR"
}

main "$@"
