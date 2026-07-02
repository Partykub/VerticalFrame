#!/usr/bin/env bash
# Batch-process all videos in /data/input, writing results to /data/output.
#
# Usage (inside container):
#   bash scripts/batch_process.sh
#
# Usage (from host):
#   docker compose run --rm --entrypoint bash verticalframe scripts/batch_process.sh
#   COMPOSE_PROFILES=cpu docker compose run --rm --entrypoint bash verticalframe-cpu scripts/batch_process.sh

set -euo pipefail

INPUT_DIR="${INPUT_DIR:-/data/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/output}"
MARKER_SUFFIX=".verticalframe.done"

mkdir -p "${OUTPUT_DIR}"

shopt -s nullglob
FILES=("${INPUT_DIR}"/*.mp4 "${INPUT_DIR}"/*.MP4 "${INPUT_DIR}"/*.mov "${INPUT_DIR}"/*.MOV \
       "${INPUT_DIR}"/*.avi "${INPUT_DIR}"/*.AVI "${INPUT_DIR}"/*.mkv "${INPUT_DIR}"/*.MKV)

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No video files found in ${INPUT_DIR}"
    exit 0
fi

PROCESSED=0
SKIPPED=0
FAILED=0

for INPUT in "${FILES[@]}"; do
    BASENAME="$(basename "${INPUT}")"
    STEM="${BASENAME%.*}"
    OUTPUT="${OUTPUT_DIR}/${STEM}-vertical.mp4"
    MARKER="${OUTPUT_DIR}/${STEM}${MARKER_SUFFIX}"

    if [[ -f "${OUTPUT}" ]] || [[ -f "${MARKER}" ]]; then
        echo "[skip] ${BASENAME} (output or marker exists)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "========================================"
    echo "[process] ${INPUT}"
    echo "          -> ${OUTPUT}"
    echo "========================================"

    if python auto_reframe.py "${INPUT}" --output "${OUTPUT}"; then
        touch "${MARKER}"
        PROCESSED=$((PROCESSED + 1))
        echo "[done] ${BASENAME}"
    else
        FAILED=$((FAILED + 1))
        echo "[fail] ${BASENAME}" >&2
        rm -f "${OUTPUT}" "${MARKER}"
    fi
done

echo ""
echo "Batch complete: processed=${PROCESSED} skipped=${SKIPPED} failed=${FAILED}"

if [[ ${FAILED} -gt 0 ]]; then
    exit 1
fi
