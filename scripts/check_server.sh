#!/usr/bin/env bash
# Pre-deploy server check for VerticalFrame.
# Run on the target Linux server before building the Docker image.
#
# Usage:
#   bash scripts/check_server.sh

set -euo pipefail

PASS=0
WARN=0
FAIL=0

ok()   { echo "  [OK]   $*"; PASS=$((PASS + 1)); }
warn() { echo "  [WARN] $*"; WARN=$((WARN + 1)); }
fail() { echo "  [FAIL] $*"; FAIL=$((FAIL + 1)); }

section() {
    echo ""
    echo "=== $* ==="
}

section "Operating System"
if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    ok "OS: ${PRETTY_NAME:-unknown}"
else
    warn "Cannot detect OS (/etc/os-release missing)"
fi
uname -a | sed 's/^/  /'

section "CPU & Memory"
if command -v nproc >/dev/null 2>&1; then
    ok "CPU cores: $(nproc)"
else
    warn "nproc not available"
fi

if [[ -f /proc/meminfo ]]; then
    MEM_KB=$(grep -E '^MemTotal:' /proc/meminfo | awk '{print $2}')
    MEM_GB=$((MEM_KB / 1024 / 1024))
    if [[ ${MEM_GB} -ge 16 ]]; then
        ok "RAM: ~${MEM_GB} GB"
    elif [[ ${MEM_GB} -ge 8 ]]; then
        warn "RAM: ~${MEM_GB} GB (16+ GB recommended for batch_size=128)"
    else
        fail "RAM: ~${MEM_GB} GB (8+ GB minimum recommended)"
    fi
else
    warn "Cannot read /proc/meminfo"
fi

section "Disk Space"
for DIR in . /var/lib/docker /tmp; do
    if [[ -d "${DIR}" ]]; then
        AVAIL=$(df -h "${DIR}" 2>/dev/null | awk 'NR==2 {print $4 " free on " $1}')
        ok "${DIR}: ${AVAIL}"
    fi
done

section "NVIDIA GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
        ok "nvidia-smi works"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/         /'
        DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        ok "Driver: ${DRIVER} (CUDA 12.8 wheels require recent driver; see DEPLOY.md)"
    else
        fail "nvidia-smi found but returned an error"
    fi
else
    warn "nvidia-smi not found — use CPU profile (COMPOSE_PROFILES=cpu) or install NVIDIA drivers"
fi

section "Docker"
if command -v docker >/dev/null 2>&1; then
    ok "Docker: $(docker --version)"
    if docker info >/dev/null 2>&1; then
        ok "Docker daemon is running"
    else
        fail "Docker daemon not reachable (permission or service stopped?)"
    fi
else
    fail "Docker not installed"
fi

section "Docker Compose"
if docker compose version >/dev/null 2>&1; then
    ok "Docker Compose: $(docker compose version --short 2>/dev/null || docker compose version)"
else
    fail "docker compose plugin not found"
fi

section "NVIDIA Container Toolkit"
if command -v nvidia-smi >/dev/null 2>&1; then
    if docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        ok "GPU passthrough works in Docker"
    else
        warn "GPU not available in Docker — install NVIDIA Container Toolkit (see DEPLOY.md)"
    fi
else
    warn "Skipped (no host GPU)"
fi

section "FFmpeg (host, optional)"
if command -v ffmpeg >/dev/null 2>&1; then
    ok "Host ffmpeg: $(ffmpeg -version 2>&1 | head -1)"
else
    warn "Host ffmpeg not installed (container includes ffmpeg)"
fi

echo ""
echo "========================================"
echo "Summary: ${PASS} passed, ${WARN} warnings, ${FAIL} failures"
echo "========================================"

if [[ ${FAIL} -gt 0 ]]; then
    echo "Fix failures before deploying. See DEPLOY.md for details."
    exit 1
fi

if [[ ${WARN} -gt 0 ]]; then
    echo "Warnings present — deployment may work with reduced performance or CPU mode."
fi

echo "Server looks ready for VerticalFrame deployment."
