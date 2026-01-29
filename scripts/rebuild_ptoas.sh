#!/usr/bin/env bash
set -euo pipefail

# Rebuild `ptoas` from a local llvm-project checkout and copy it into this repo.
#
# Usage:
#   LLVM_PROJECT_DIR=$HOME/llvm-project BUILD_DIR=$HOME/llvm-project/build \
#     ./scripts/rebuild_ptoas.sh
#
# Notes:
# - The destination is chosen by host OS/arch (same mapping as `bin/ptoas`).
# - This script assumes you already configured a build that includes the PTO dialect/tools.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

llvm_project_dir="${LLVM_PROJECT_DIR:-$HOME/llvm-project}"
build_dir="${BUILD_DIR:-}"

if [[ -z "${build_dir}" ]]; then
  # Prefer a conventional `${llvm_project_dir}/build`, but auto-detect a build dir
  # by searching for `bin/ptoas` (keeps the script usable across different local
  # build folder names).
  if [[ -d "${llvm_project_dir}/build" ]]; then
    build_dir="${llvm_project_dir}/build"
  else
    found="$(find "${llvm_project_dir}" -maxdepth 2 -type f -path "*/bin/ptoas" 2>/dev/null | head -n 1 || true)"
    if [[ -n "${found}" ]]; then
      build_dir="$(dirname "$(dirname "${found}")")"
    else
      build_dir="${llvm_project_dir}/build"
    fi
  fi
fi

os="$(uname -s)"
arch="$(uname -m)"
case "${os}:${arch}" in
  Linux:aarch64|Linux:arm64) subdir="linux-aarch64" ;;
  Linux:x86_64|Linux:amd64) subdir="linux-x86_64" ;;
  Darwin:arm64) subdir="macos-aarch64" ;;
  *)
    echo "error: unsupported platform: ${os} ${arch}" >&2
    exit 2
    ;;
esac

src="${build_dir}/bin/ptoas"
dst="${repo_root}/bin/${subdir}/ptoas"

if [[ ! -d "${llvm_project_dir}" ]]; then
  echo "error: LLVM_PROJECT_DIR not found: ${llvm_project_dir}" >&2
  exit 2
fi
if [[ ! -d "${build_dir}" ]]; then
  echo "error: BUILD_DIR not found: ${build_dir}" >&2
  exit 2
fi

echo "[rebuild_ptoas] building: ${src}"
ninja -C "${build_dir}" ptoas

if [[ ! -x "${src}" ]]; then
  echo "error: build produced no executable at: ${src}" >&2
  exit 2
fi

mkdir -p "$(dirname "${dst}")"
cp -f "${src}" "${dst}"
chmod +x "${dst}"

echo "[rebuild_ptoas] updated: ${dst}"
file "${dst}" || true

tarball="${dst}.tar.gz"
echo "[rebuild_ptoas] packaging: ${tarball}"
tar -czf "${tarball}" -C "$(dirname "${dst}")" "$(basename "${dst}")"
bytes="$(wc -c <"${tarball}" | tr -d ' ')"
limit="$((100 * 1024 * 1024))"
if [[ "${bytes}" =~ ^[0-9]+$ ]] && (( bytes > limit )); then
  echo "error: tarball exceeds 100MB: ${tarball} (${bytes} bytes)" >&2
  exit 2
fi
ls -lah "${tarball}" || true
