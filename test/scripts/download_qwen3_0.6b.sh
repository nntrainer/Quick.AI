#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# @file   download_qwen3_0.6b.sh
# @date   24 Apr 2026
# @brief  Fetch the Qwen3-0.6B Q4_0 weights from
#         github.com/eunjuyang/nntrainer-causallm-models, reassemble the
#         split .bin parts via combine.sh, verify the SHA256 manifest and
#         stage the directory under ./models/ where the Quick.AI C API
#         expects to find it.
# @see    https://github.com/eunjuyang/nntrainer-causallm-models
# @author Eunju Yang <ej.yang@samsung.com>
#
# Usage:
#   test/scripts/download_qwen3_0.6b.sh [install_root]
#
#   install_root defaults to the repo root. The script writes the model
#   into "$install_root/models/qwen3-0.6b-w16a16/" because the C API
#   resolve_model_path() maps (QWEN3-0.6B, W16A16) to that directory and
#   W16A16 is unregistered in the built-in config table, which forces the
#   API onto the external-JSON path (CASE 2) that reads the nntr_config.json
#   shipped with the model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INSTALL_ROOT="${1:-$REPO_ROOT}"

MODEL_REPO_URL="https://github.com/eunjuyang/nntrainer-causallm-models.git"
MODEL_REPO_BRANCH="main"
MODEL_DIR_NAME="qwen3-0.6b-q40-x86"
TARGET_DIR_NAME="qwen3-0.6b-w16a16"

CACHE_DIR="$INSTALL_ROOT/.test_cache/nntrainer-causallm-models"
TARGET_DIR="$INSTALL_ROOT/models/$TARGET_DIR_NAME"

log() { echo "[download_qwen3_0.6b] $*"; }

ensure_clone() {
  if [ -d "$CACHE_DIR/.git" ]; then
    log "model cache exists at $CACHE_DIR; fetching latest $MODEL_REPO_BRANCH"
    git -C "$CACHE_DIR" fetch --depth 1 origin "$MODEL_REPO_BRANCH"
    git -C "$CACHE_DIR" checkout -q "$MODEL_REPO_BRANCH"
    git -C "$CACHE_DIR" reset --hard "origin/$MODEL_REPO_BRANCH"
  else
    log "cloning $MODEL_REPO_URL -> $CACHE_DIR"
    mkdir -p "$(dirname "$CACHE_DIR")"
    git clone --depth 1 --branch "$MODEL_REPO_BRANCH" "$MODEL_REPO_URL" "$CACHE_DIR"
  fi
}

combine_bin() {
  local src="$CACHE_DIR/$MODEL_DIR_NAME"
  if [ ! -f "$src/combine.sh" ]; then
    log "ERROR: combine.sh not found under $src"
    exit 1
  fi
  log "running combine.sh to reassemble the split .bin"
  chmod +x "$src/combine.sh"
  ( cd "$src" && ./combine.sh )
}

verify_sha256() {
  local src="$CACHE_DIR/$MODEL_DIR_NAME"
  if [ -f "$src/SHA256SUMS" ]; then
    log "verifying SHA256SUMS for config + tokenizer assets"
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

  # Copy JSON/text configs and the reassembled bin. Skip the split parts so
  # the install directory stays lean.
  for f in config.json generation_config.json nntr_config.json \
           tokenizer.json tokenizer_config.json vocab.json merges.txt \
           nntr_qwen3_0.6b_w4e4a32.bin; do
    if [ -f "$src/$f" ]; then
      cp "$src/$f" "$TARGET_DIR/"
    fi
  done
}

main() {
  ensure_clone
  combine_bin
  verify_sha256
  install_to_target
  log "done. QUICKAI_TEST_MODEL_DIR=$TARGET_DIR"
  echo "$TARGET_DIR"
}

main "$@"
