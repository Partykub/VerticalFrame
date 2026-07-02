# VerticalFrame deployment guide (IT / ops)

Short reference for running VerticalFrame on a Linux server (or locally via Docker Desktop).

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **OS** | Ubuntu 22.04+ or Debian 12+ recommended; Windows/macOS via Docker Desktop for dev |
| **Docker** | Engine 24+ with Compose v2 (`docker compose`) |
| **RAM** | 8 GB minimum; 16+ GB recommended (`batch_size: 128` in `config.json`) |
| **Disk** | ~8–15 GB for image + models; extra space for input/output videos |
| **GPU (optional)** | NVIDIA GPU + driver supporting CUDA 12.8 wheels |
| **NVIDIA Container Toolkit** | Required only for GPU mode (Linux) or WSL2 backend on Windows |

### Install NVIDIA Container Toolkit (GPU servers)

Follow the official guide:  
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

After install, verify:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## AI stack (what gets installed)

| Component | Role | Package / model |
|-----------|------|-----------------|
| **YOLOv8m** | Body detection | `yolov8m.pt` |
| **YOLOv8m-face** | Face detection | `yolov8m-face.pt` |
| **Spectral Saliency** | Fallback focus when no people | `opencv-contrib-python` |
| **DeepSORT** | Multi-person tracking (stable track IDs) | `deep-sort-realtime` |
| **OpenAI CLIP (RN50)** | Re-ID embedder for DeepSORT | `git+https://github.com/openai/CLIP.git` |
| **FFmpeg** | Render + audio merge | System package in image |

### Why OpenAI CLIP?

Default config uses DeepSORT with `"embedder": "clip_RN50"`. CLIP converts each detected person into a visual “fingerprint” so the tracker can tell whether two boxes in different frames are the **same person** — reducing ID swaps when people overlap or briefly leave frame.

| Embedder | Speed | Tracking quality |
|----------|-------|------------------|
| `mobilenet` | Faster | Good |
| **`clip_RN50`** (default) | Slower | Better in crowds / occlusions |

To drop CLIP (faster install, lighter deps), set in `config.json`:

```json
"tracking": {
  "deepsort": {
    "embedder": "mobilenet"
  }
}
```

Then remove the CLIP line from `requirements-gpu.txt` and rebuild.

## Pre-deploy check

On the server, clone the repo and run:

```bash
bash scripts/check_server.sh
```

## Build

```bash
cd VerticalFrame
docker compose build
```

- **First build:** ~30–45 minutes (PyTorch CUDA wheels + model download).
- **Later builds:** faster when Docker layer cache is warm.
- Models baked into the image: `yolov8m.pt`, `yolov8m-face.pt`, DeepSORT CLIP warmup.

## Directory layout

```
VerticalFrame/
├── config.json          # mount read-only into container
├── data/
│   ├── input/           # place source videos here
│   └── output/          # processed videos appear here
└── docker-compose.yml
```

## CLI — single video

**GPU (default):**

```bash
docker compose run --rm verticalframe \
  /data/input/video.mp4 \
  --output /data/output/video-vertical.mp4
```

**Helper script (from host, Linux/macOS/Git Bash):**

```bash
chmod +x scripts/run.sh
./scripts/run.sh /data/input/video.mp4
```

**Debug view (side-by-side):**

```bash
docker compose run --rm verticalframe \
  /data/input/video.mp4 \
  --debug-view --output /data/output/video-debug.mp4
```

**CPU-only server:**

```bash
COMPOSE_PROFILES=cpu docker compose run --rm verticalframe-cpu \
  /data/input/video.mp4 \
  --output /data/output/video-vertical.mp4
```

## Test on your machine (Docker Desktop)

Verified on Windows 11 + Docker Desktop + NVIDIA GPU:

```powershell
cd VerticalFrame
docker compose build
Copy-Item "C:\path\to\video.mp4" "data\input\"
docker compose run --rm verticalframe `
  /data/input/video.mp4 `
  --output /data/output/video-vertical.mp4
```

Check logs for `cuda:0` during model load. Output appears in `data\output\`.

## Batch mode

1. Copy videos into `data/input/`
2. Run batch script inside the container:

```bash
cp /path/to/clips/*.mp4 data/input/
docker compose run --rm --entrypoint bash verticalframe scripts/batch_process.sh
```

Batch skips files that already have an output (`*-vertical.mp4`) or a `.verticalframe.done` marker in `data/output/`.

## Configuration

Edit `config.json` on the host (mounted read-only). Common tuning:

| Key | Effect |
|-----|--------|
| `scanner.batch_size` | Lower (e.g. 32) if CUDA out-of-memory |
| `tracking.deepsort.embedder` | `clip_RN50` (default) or `mobilenet` (faster, no CLIP) |
| `scanner.yolo_face_model` | Default `yolov8m-face`; use `yolov8n-face` for speed |
| `scanner.face_detector` | `yolo` (GPU) or `mediapipe` (CPU) |

## Dependencies (native install)

Split intentionally — do **not** use `opencv-python` alone (saliency module requires contrib):

```bash
pip install -r requirements.txt
pip install -r requirements-gpu.txt   # torch cu128, opencv-contrib, deep-sort, CLIP
python scripts/download_models.py
```

| File | Contents |
|------|----------|
| `requirements.txt` | Core: ultralytics, mediapipe, scipy, … |
| `requirements-gpu.txt` | PyTorch cu128, opencv-contrib-python, deep-sort-realtime, OpenAI CLIP |
| `requirements-dev.txt` | pytest (optional) |

## Troubleshooting

### CUDA out of memory

Reduce `scanner.batch_size` in `config.json` (try 64 or 32), then re-run.

### `cuda:0` not used / runs on CPU

- Confirm `nvidia-smi` works on the host
- Confirm NVIDIA Container Toolkit is installed (Linux) or GPU enabled in Docker Desktop (Windows)
- Use the default `verticalframe` service (not `verticalframe-cpu`)
- Check logs for `cuda:0` during model load

### Docker build fails on CLIP / git

The image installs `git` and pulls CLIP from GitHub during `pip install`. Ensure the build host has network access. Alternative: switch embedder to `mobilenet` and remove CLIP from `requirements-gpu.txt`.

### Driver too old for CUDA 12.8

If PyTorch fails to load CUDA, either upgrade the NVIDIA driver or change the CUDA base image and `requirements-gpu.txt` index to match your driver (e.g. `cu124`).

### No audio in output

Normal if the source video has no audio track. The pipeline merges audio via ffmpeg when present.

### Permission errors on `data/output`

```bash
chmod 777 data/output   # or chown to your deploy user
```

### Re-download models manually

```bash
docker compose run --rm --entrypoint python3 verticalframe scripts/download_models.py
```

## Local development (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\Activate.ps1                        # Windows

pip install -r requirements.txt
pip install -r requirements-gpu.txt
python scripts/download_models.py
python auto_reframe.py input.mp4 --output output.mp4
```

For tests: `pip install -r requirements-dev.txt && pytest`
