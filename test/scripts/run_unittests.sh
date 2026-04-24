#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# @file   run_unittests.sh
# @date   24 Apr 2026
# @brief  Stages the Qwen3-0.6B model, reconfigures the meson build with
#         -Denable-test=true and runs the full GoogleTest suite.
# @author Eunju Yang <ej.yang@samsung.com>
#
# Usage:
#   test/scripts/run_unittests.sh [build_dir]
#
# Environment:
#   QUICKAI_SKIP_MODEL_DOWNLOAD=1   skip the qwen3-0.6b download step
#                                   (tests that need weights will be skipped)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${1:-$REPO_ROOT/build}"

log() { echo "[run_unittests] $*"; }

stage_model() {
  if [ "${QUICKAI_SKIP_MODEL_DOWNLOAD:-0}" = "1" ]; then
    log "QUICKAI_SKIP_MODEL_DOWNLOAD=1, skipping model download"
    return
  fi
  if [ ! -f "$REPO_ROOT/models/qwen3-0.6b-w16a16/nntr_config.json" ]; then
    log "staging Qwen3-0.6B Q4_0 weights"
    "$SCRIPT_DIR/download_qwen3_0.6b.sh" "$REPO_ROOT"
  else
    log "Qwen3-0.6B weights already present, skipping download"
  fi
}

configure_build() {
  if [ ! -f "$BUILD_DIR/build.ninja" ] && [ ! -f "$BUILD_DIR/meson-info/intro-buildoptions.json" ]; then
    log "meson setup $BUILD_DIR"
    meson setup "$BUILD_DIR" "$REPO_ROOT" -Denable-test=true
  else
    log "meson reconfigure $BUILD_DIR (enable-test=true)"
    meson configure "$BUILD_DIR" -Denable-test=true
  fi
}

build_and_test() {
  log "ninja -C $BUILD_DIR"
  ninja -C "$BUILD_DIR"

  log "meson test -C $BUILD_DIR --print-errorlogs"
  meson test -C "$BUILD_DIR" --print-errorlogs
}

main() {
  stage_model
  configure_build
  build_and_test
  log "all tests completed"
}

main "$@"
