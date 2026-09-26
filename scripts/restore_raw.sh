#!/usr/bin/env bash
set -euo pipefail

readonly RELEASE_REPO="tarudesu/viLens-concepts-preparation"
readonly RELEASE_TAG="v1.2-data"
readonly ARCHIVES=(
  "viLens-cache-v1.2.tar.zst"
  "viLens-raw-small-v1.2.tar.zst"
  "viLens-raw-wiktextract-v1.2.tar.zst"
)

fail() {
  printf 'restore_raw.sh: %s\n' "$*" >&2
  exit 1
}

for command_name in gh zstd tar; do
  command -v "$command_name" >/dev/null 2>&1 || fail "required command not found: $command_name"
done
if command -v sha256sum >/dev/null 2>&1; then
  checksum_command="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
  checksum_command="shasum"
else
  fail "sha256sum or shasum is required"
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
umask 077
temp_root="$(mktemp -d "${TMPDIR:-/tmp}/vilens-raw.XXXXXX")"
trap 'rm -rf "$temp_root"' EXIT
download_dir="$temp_root/download"
mkdir -p "$download_dir" "$repo_root/data/raw"

gh release download "$RELEASE_TAG" -R "$RELEASE_REPO" --dir "$download_dir"
[[ -s "$download_dir/SHA256SUMS" ]] || fail "release is missing SHA256SUMS"
cmp -s "$download_dir/SHA256SUMS" "$repo_root/docs/data-archive/SHA256SUMS" \
  || fail "release checksum manifest differs from docs/data-archive/SHA256SUMS"
(
  cd "$download_dir"
  if [[ "$checksum_command" == "sha256sum" ]]; then
    sha256sum --check SHA256SUMS
  else
    shasum -a 256 --check SHA256SUMS
  fi
)

for archive_name in "${ARCHIVES[@]}"; do
  archive_path="$download_dir/$archive_name"
  if [[ ! -f "$archive_path" ]]; then
    parts=( "$download_dir/$archive_name.part-"* )
    [[ -f "${parts[0]}" ]] || fail "release is missing $archive_name or its split parts"
    archive_path="$temp_root/$archive_name"
    cat "${parts[@]}" > "$archive_path"
  fi

  zstd --test "$archive_path" >/dev/null || fail "archive failed zstd integrity check: $archive_name"
  listing="$temp_root/${archive_name}.list"
  zstd --decompress --stdout "$archive_path" | tar -tf - > "$listing"
  while IFS= read -r member; do
    case "$member" in
      data/raw/*) ;;
      *) fail "archive contains an unexpected path: $member" ;;
    esac
    case "/$member/" in
      */../*) fail "archive contains a parent traversal path: $member" ;;
    esac
  done < "$listing"
  zstd --decompress --stdout "$archive_path" | tar -xf - -C "$repo_root"
  printf 'Restored %s\n' "$archive_name"
done
