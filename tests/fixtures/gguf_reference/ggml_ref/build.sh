#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Builds ggml_ref (ref.c) against ggml at a pinned llama.cpp commit.
#
# Usage: tests/fixtures/gguf_reference/ggml_ref/build.sh
#   Prints the path of the built binary on its last line.
#   Needs git, cmake, gcc and g++ (Linux or WSL; it is how the Phase 7.10
#   fixtures were made, under WSL2 Ubuntu 24.04).
#
# The checkout and build live OUTSIDE the repository, in $GGML_REF_DIR
# (default ~/.cache/anamnesis-ggml-ref), so nothing here is ever packaged.
#
# GGML_NATIVE=OFF matters: with it on, the compiler may use FMA and contract
# `a * b + c`, and a golden would then depend on the build host's CPU. Off, the
# reference is the portable C exactly as written.
set -euo pipefail

COMMIT=37b53fd4545847188fdad29e38ba57875efc8228
OUT="${GGML_REF_DIR:-$HOME/.cache/anamnesis-ggml-ref}"
SRC="$OUT/llama.cpp"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$SRC/.git" ]; then
  git init -q "$SRC"
  git -C "$SRC" remote add origin https://github.com/ggml-org/llama.cpp
fi
if [ "$(git -C "$SRC" rev-parse -q --verify HEAD 2>/dev/null || true)" != "$COMMIT" ]; then
  git -C "$SRC" fetch -q --depth 1 origin "$COMMIT"
  git -C "$SRC" checkout -q --detach "$COMMIT"
fi

# Configured from llama.cpp's top level: ggml/ inside llama.cpp is not a
# standalone CMake project (it expects the ggml.pc.in that the separate ggml
# repository ships). Only the ggml-base target is built.
cmake -S "$SRC" -B "$OUT/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_BACKEND_DL=OFF \
  -DGGML_OPENMP=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_CURL=OFF > "$OUT/cmake.log"
cmake --build "$OUT/build" --target ggml-base -j > "$OUT/build.log"

lib="$(find "$OUT/build" -name 'libggml-base.a' | head -n 1)"
gcc -O2 -std=c11 -I "$SRC/ggml/include" -c "$HERE/ref.c" -o "$OUT/ref.o"
g++ "$OUT/ref.o" "$lib" -lm -lpthread -o "$OUT/ggml_ref"

echo "ggml_ref built against llama.cpp $COMMIT"
echo "$OUT/ggml_ref"
