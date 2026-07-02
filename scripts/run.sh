#!/usr/bin/env bash
# Wrapper for single-clip CLI via Docker Compose.
#
# Usage:
#   ./scripts/run.sh /data/input/video.mp4
#   ./scripts/run.sh /data/input/video.mp4 --debug-view
#   ./scripts/run.sh /data/input/video.mp4 --output /data/output/custom.mp4
#
# CPU-only server:
#   COMPOSE_PROFILES=cpu ./scripts/run.sh /data/input/video.mp4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <input_video> [auto_reframe.py options...]" >&2
    echo "Example: $0 /data/input/clip.mp4 --debug-view" >&2
    exit 1
fi

INPUT="$1"
shift

BASENAME="$(basename "${INPUT}")"
STEM="${BASENAME%.*}"
DEFAULT_OUTPUT="/data/output/${STEM}-vertical.mp4"

# Use default output unless caller passes --output
HAS_OUTPUT=false
for arg in "$@"; do
    if [[ "${arg}" == "--output" ]]; then
        HAS_OUTPUT=true
        break
    fi
done

SERVICE="verticalframe"
if [[ "${COMPOSE_PROFILES:-}" == "cpu" ]]; then
    SERVICE="verticalframe-cpu"
fi

if [[ "${HAS_OUTPUT}" == true ]]; then
    docker compose run --rm "${SERVICE}" "${INPUT}" "$@"
else
    docker compose run --rm "${SERVICE}" "${INPUT}" --output "${DEFAULT_OUTPUT}" "$@"
fi
