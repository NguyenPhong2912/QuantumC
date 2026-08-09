#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DOWNLOAD_ROOT="$PROJECT_ROOT/data/downloads/bdd100k"
RAW_ROOT="$PROJECT_ROOT/data/raw/bdd100k"

BASE_URL="https://archive.org/download/bdd100k"
IMAGES_NAME="bdd100k_images.zip"
LABELS_NAME="bdd100k_labels.zip"
IMAGES_SHA1="8678f14548c8a6e2f7de3183f16d34c7f1e1f37a"
LABELS_SHA1="d45200a65ace8ab034d0caed6b09fa3b4735f4b9"

mkdir -p "$DOWNLOAD_ROOT" "$RAW_ROOT"

download_and_verify() {
    local name="$1"
    local expected_sha1="$2"
    local target="$DOWNLOAD_ROOT/$name"

    if [[ -f "$target" ]]; then
        local current_sha1
        current_sha1="$(sha1sum "$target" | awk '{print $1}')"
        if [[ "$current_sha1" == "$expected_sha1" ]]; then
            echo "$name already verified"
            return
        fi
        echo "$name checksum mismatch; resuming/replacing download"
    fi

    curl \
        --fail \
        --location \
        --retry 10 \
        --retry-all-errors \
        --continue-at - \
        --output "$target" \
        "$BASE_URL/$name"

    echo "$expected_sha1  $target" | sha1sum --check --status
    unzip -tq "$target" >/dev/null
    echo "$name download and archive verification passed"
}

download_and_verify "$IMAGES_NAME" "$IMAGES_SHA1"
download_and_verify "$LABELS_NAME" "$LABELS_SHA1"

IMAGES_MARKER="$RAW_ROOT/.images-extracted-$IMAGES_SHA1"
LABELS_MARKER="$RAW_ROOT/.labels-extracted-$LABELS_SHA1"

if [[ ! -f "$IMAGES_MARKER" ]]; then
    unzip -oq "$DOWNLOAD_ROOT/$IMAGES_NAME" -d "$RAW_ROOT"
    touch "$IMAGES_MARKER"
fi

if [[ ! -f "$LABELS_MARKER" ]]; then
    unzip -oq "$DOWNLOAD_ROOT/$LABELS_NAME" -d "$RAW_ROOT"
    touch "$LABELS_MARKER"
fi

echo "BDD100K MIRROR DOWNLOAD PASSED"
