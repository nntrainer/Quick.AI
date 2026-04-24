#!/bin/bash

# Build script for CausalLM Android application
# This script builds libquick_dot_ai_core.so and quick_dot_ai executable
set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN} $1 ${NC}"
    echo -e "${CYAN}========================================${NC}"
}

log_step() {
    echo -e "\n${YELLOW}[Step $1]${NC} $2"
    echo -e "${YELLOW}----------------------------------------${NC}"
}

# Function to check and fix artifact location
check_artifact() {
    local filename=$1
    local libs_path="libs/arm64-v8a/$filename"
    local obj_path="obj/local/arm64-v8a/$filename"

    if [ -f "$libs_path" ]; then
        size=$(ls -lh "$libs_path" | awk '{print $5}')
        echo -e "  ${GREEN}[OK]${NC} $filename ($size)"
        return 0
    elif [ -f "$obj_path" ]; then
        echo -e "  ${YELLOW}[WARN]${NC} $filename found in obj but not in libs. Copying..."
        mkdir -p "libs/arm64-v8a"
        cp "$obj_path" "$libs_path"
        if [ -x "$obj_path" ]; then
            chmod +x "$libs_path"
        fi
        size=$(ls -lh "$libs_path" | awk '{print $5}')
        echo -e "  ${GREEN}[OK]${NC} $filename ($size) (Copied from obj)"
        return 0
    else
        echo -e "  ${RED}[ERROR]${NC} $filename not found!"
        log_info "  Checked paths:"
        log_info "    - $libs_path"
        log_info "    - $obj_path"
        return 1
    fi
}

# Check if NDK path is set
if [ -z "$ANDROID_NDK" ]; then
    log_error "ANDROID_NDK is not set. Please set it to your Android NDK path."
    log_info "Example: export ANDROID_NDK=/path/to/android-ndk-r21d"
    exit 1
fi

# Set NNTRAINER_ROOT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="${NNTRAINER_ROOT:-$(cd "$SCRIPT_DIR/subprojects/nntrainer" && pwd)}"
export NNTRAINER_ROOT

log_header "Build CausalLM Android Application"
log_info "NNTRAINER_ROOT: $NNTRAINER_ROOT"
log_info "ANDROID_NDK: $ANDROID_NDK"
log_info "Working directory: $(pwd)"

# Step 1: Build nntrainer for Android if not already built
log_step "1/4" "Build nntrainer for Android"

if [ ! -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
    log_info "Building nntrainer for Android..."
    cd "$NNTRAINER_ROOT"
    if [ -d "$NNTRAINER_ROOT/builddir" ]; then
        rm -rf builddir
    fi
    # Use a user-writable prefix so `meson install` does not try to
    # escalate privileges to write into /usr/local (fails in CI / non-
    # interactive environments).
    ./tools/package_android.sh "-Dprefix=$NNTRAINER_ROOT/builddir/install"
else
    log_info "nntrainer for Android already built (skipping)"
fi

# Check if build was successful
if [ ! -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
    log_error "nntrainer build failed. Please check the build logs."
    exit 1
fi
log_success "nntrainer ready"

# Step 2: Build tokenizer library if not present
log_step "2/4" "Build Tokenizer Library"

cd "$SCRIPT_DIR"
if [ ! -f "lib/libtokenizers_android_c.a" ]; then
    log_warning "libtokenizers_android_c.a not found in lib directory."
    log_info "Attempting to build tokenizer library..."
    if [ -f "build_tokenizer_android.sh" ]; then
        ./build_tokenizer_android.sh
    else
        log_error "tokenizer library not found and build script is missing."
        log_info "Please build or download the tokenizer library for Android arm64-v8a"
        log_info "and place it in: $SCRIPT_DIR/lib/libtokenizers_android_c.a"
        exit 1
    fi
else
    log_info "Tokenizer library already built (skipping)"
fi
log_success "Tokenizer library ready"

# Step 3: Prepare json.hpp if not present
log_step "3/4" "Prepare json.hpp"

if [ ! -f "$SCRIPT_DIR/json.hpp" ]; then
    log_info "json.hpp not found. Downloading..."
    # Use Quick.AI's own prepare_encoder.sh so the downloaded json.hpp
    # is copied to the Quick.AI project root (the nntrainer submodule
    # ships a legacy variant that drops it into Applications/CausalLM/
    # instead and leaves our expected location empty).
    "$SCRIPT_DIR/jni/prepare_encoder.sh" "$NNTRAINER_ROOT/builddir" "0.2"

    if [ ! -f "$SCRIPT_DIR/json.hpp" ]; then
        log_error "Failed to download json.hpp"
        exit 1
    fi
else
    log_info "json.hpp already exists (skipping)"
fi
log_success "json.hpp ready"

# Step 4: Build CausalLM (libquick_dot_ai_core.so and quick_dot_ai)
log_step "4/4" "Build CausalLM Core (library + executable)"

cd "$SCRIPT_DIR/jni"

# Clean previous builds
rm -rf libs obj

log_info "Building with ndk-build (builds quick_dot_ai_core, quick_dot_ai, quick_dot_ai_quantize)..."
# We explicitly set paths to ensure outputs are predictable
if ndk-build NDK_PROJECT_PATH=. NDK_LIBS_OUT=./libs NDK_OUT=./obj APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk quick_dot_ai_core quick_dot_ai  quick_dot_ai_quantize -j $(nproc); then
    log_success "Build completed successfully"
else
    log_error "Build failed"
    exit 1
fi

# Verify outputs
log_info "Build artifacts:"

check_artifact "libquick_dot_ai_core.so" || exit 1
check_artifact "quick_dot_ai" || exit 1
check_artifact "quick_dot_ai_quantize" || exit 1

# Summary
log_header "Build Summary"
log_success "Build completed successfully!"
log_info "Output files are in: $SCRIPT_DIR/jni/libs/arm64-v8a/"
log_info "Executables:"
log_info "  - quick_dot_ai (main application), quick_dot_ai_quantize"
log_info "Libraries:"
log_info "  - libquick_dot_ai_core.so (CausalLM Core library)"
log_info "  - libnntrainer.so (nntrainer library)"
log_info "  - libccapi-nntrainer.so (nntrainer C/C API)"
log_info "  - libc++_shared.so (C++ runtime)"
log_info "To build API library, run:"
log_info "  ./build_api_lib.sh"
log_info "To install and run:"
log_info "  ./install_android.sh"
