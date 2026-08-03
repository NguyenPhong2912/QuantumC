#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
)"

DOWNLOAD_DIR="${PROJECT_ROOT}/data/downloads/voc"
RAW_ROOT="${PROJECT_ROOT}/data/raw/voc"
EXTRACT_ROOT="${RAW_ROOT}/extracted"
MARKER="${EXTRACT_ROOT}/.extraction_complete"

ARCHIVES=(
  "VOCtrainval_06-Nov-2007.zip"
  "VOCtest_06-Nov-2007.zip"
  "VOCtrainval_11-May-2012.zip"
)

mkdir -p "${RAW_ROOT}"

if [[ -f "${MARKER}" ]]; then
  echo "VOC extraction already completed:"
  cat "${MARKER}"
  exit 0
fi

for archive_name in "${ARCHIVES[@]}"; do
  archive="${DOWNLOAD_DIR}/${archive_name}"

  if [[ ! -s "${archive}" ]]; then
    echo "ERROR: missing archive: ${archive}"
    exit 1
  fi
done

echo "Verifying archive checksums..."

(
  cd "${DOWNLOAD_DIR}"
  sha256sum --check SHA256SUMS
)

if [[ -d "${EXTRACT_ROOT}" ]]; then
  echo "Removing incomplete extraction directory:"
  echo "${EXTRACT_ROOT}"
  rm -rf "${EXTRACT_ROOT}"
fi

mkdir -p "${EXTRACT_ROOT}"

for archive_name in "${ARCHIVES[@]}"; do
  archive="${DOWNLOAD_DIR}/${archive_name}"

  echo
  echo "===== EXTRACTING: ${archive_name} ====="

  unzip \
    -q \
    -o \
    "${archive}" \
    -d "${EXTRACT_ROOT}"
done

VOC2007="${EXTRACT_ROOT}/VOCdevkit/VOC2007"
VOC2012="${EXTRACT_ROOT}/VOCdevkit/VOC2012"

required_directories=(
  "${VOC2007}/Annotations"
  "${VOC2007}/ImageSets/Main"
  "${VOC2007}/JPEGImages"
  "${VOC2012}/Annotations"
  "${VOC2012}/ImageSets/Main"
  "${VOC2012}/JPEGImages"
)

for directory in "${required_directories[@]}"; do
  if [[ ! -d "${directory}" ]]; then
    echo "ERROR: required directory is missing:"
    echo "${directory}"
    exit 1
  fi
done

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "VOC2007=${VOC2007}"
  echo "VOC2012=${VOC2012}"
} > "${MARKER}"

echo
echo "===== EXTRACTION COMPLETE ====="
cat "${MARKER}"
