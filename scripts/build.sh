#!/usr/bin/env bash
# SPDX-License-Identifier: Unlicense
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cuda_compiler="${CUDACXX:-}"
if [ -z "$cuda_compiler" ]; then
  cuda_compiler="$(command -v nvcc || true)"
fi
if [ -z "$cuda_compiler" ]; then
  for candidate in /usr/local/cuda/bin/nvcc /opt/cuda/bin/nvcc; do
    if [ -x "$candidate" ]; then cuda_compiler="$candidate"; break; fi
  done
fi
if [ -z "$cuda_compiler" ]; then
  printf '%s\n' 'CUDA nvcc not found. Install CUDA >= 13.1 and set CUDACXX.' >&2
  exit 1
fi
cmake -S "$project_dir" -B "$project_dir/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_COMPILER="$cuda_compiler" "$@"
cmake --build "$project_dir/build" -j "${BUILD_JOBS:-4}"
