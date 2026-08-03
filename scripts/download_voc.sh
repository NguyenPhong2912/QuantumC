#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
)"

DOWNLOAD_DIR="${PROJECT_ROOT}/data/downloads/voc"
mkdir -p "${DOWNLOAD_DIR}"

BASE_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0"

FILES=(
  "VOCtrainval_06-Nov-2007.zip"
  "VOCtest_06-Nov-2007.zip"
  "VOCtrainval_11-May-2012.zip"
)

MINIMUM_BYTES=1000000

for filename in "${FILES[@]}"; do
  destination="${DOWNLOAD_DIR}/${filename}"
  url="${BASE_URL}/${filename}"

  if [[ -f "${destination}" ]]; then
    current_size="$(stat -c '%s' "${destination}")"

    if (( current_size >= MINIMUM_BYTES )); then
      echo "EXISTS: ${destination} (${current_size} bytes)"
      continue
    fi

    echo "REMOVING INVALID FILE: ${destination}"
    rm -f "${destination}"
  fi

  echo "DOWNLOADING: ${filename}"
  echo "SOURCE     : ${url}"

  curl \
    --fail \
    --location \
    --retry 5 \
    --retry-delay 5 \
    --retry-all-errors \
    --continue-at - \
    --output "${destination}" \
    "${url}"

  downloaded_size="$(stat -c '%s' "${destination}")"

  if (( downloaded_size < MINIMUM_BYTES )); then
    echo "ERROR: archive is unexpectedly small"
    echo "FILE : ${destination}"
    echo "SIZE : ${downloaded_size}"
    exit 1
  fi

  echo "DOWNLOADED: ${downloaded_size} bytes"
done

echo
echo "===== DOWNLOAD INVENTORY ====="
ls -lh "${DOWNLOAD_DIR}"
