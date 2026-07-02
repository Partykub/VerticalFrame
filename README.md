# Auto-Reframe Pipeline (High-Fidelity Edition)

An intelligent, AI-powered automated video reframing tool redesigned for **Professional Broadcasting Standards**. It crops horizontal videos (16:9) into vertical (9:16) formats using advanced computer vision and strictly **Lossless** processing pipelines.

## 🌟 Professional Quality Guarantee ("The Pipeline")

This project is not just a cropper; it's a **High-Fidelity Rendering Engine**:
1.  **Zero-Loss In-Memory Processing**: Frames are passed directly from OpenCV to AI models in RAM (Raw Pixel Matrices). No intermediate files, no compression artifacts.
2.  **Direct FFmpeg Piping**: We bypass standard writers and pipe raw data deeply into **FFmpeg's libx264** encoder.
3.  **Visually Lossless Encoding**:
    *   **CRF 18**: Studio-grade constant rate factor.
    *   **Preset Slow**: High-efficiency compression without quality sacrifice.
    *   **Bitrate Booster**: Output files often have higher bitrates than originals (~3x) to preserve every detail during the crop.

---

## 🚀 Key Features

*   **Multi-Stage AI**: Hybrid tracking using **YOLOv8** (People), **YOLO/MediaPipe** (Faces), and **Spectral Residual Saliency** (Attention).
*   **Smart Director**: A decision engine that prioritizes `Face > Body > Saliency` based on stability and size.
*   **DeepSORT + CLIP**: Multi-person tracking with **OpenAI CLIP** re-identification to keep stable actor IDs in crowded scenes.
*   **Cinematic Camera**:
    *   **Sine-In-Out Easing**: Organic start/stop camera movements (no robotic jerks).
    *   **Look-Ahead Logic**: The AI "sees the future" to prepare for subject movement before it happens.
    *   **Smart Lock**: Locks onto a specific "Actor ID" to prevent camera jumping in crowds.
*   **Vertical-First**: Default output is a clean, broadcast-ready 9:16 video.

---

## 🧠 AI stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Body | YOLOv8m | Detect people |
| Face | YOLOv8m-face (default) | Precise face boxes on GPU |
| Saliency | OpenCV Spectral Residual | Focus when no people in frame |
| Tracking | DeepSORT | Assign persistent track IDs |
| Re-ID | **OpenAI CLIP (RN50)** | Match the same person across frames |

**CLIP** is not used for text understanding here — only its image encoder produces a visual fingerprint so DeepSORT does not swap IDs when people cross or briefly disappear.

---

## 📦 Installation

### Prerequisites
1.  **Python 3.8+** (3.11 recommended)
2.  **FFmpeg** (CRITICAL: The core engine)
    *   *Ubuntu/WSL*: `sudo apt install ffmpeg`
    *   *Windows*: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.
3.  **NVIDIA GPU + CUDA** (recommended for production speed)

### Setup (native / GPU)

Install **both** requirement files — `opencv-contrib-python` (not `opencv-python`) is required for saliency detection:

```bash
pip install -r requirements.txt
pip install -r requirements-gpu.txt
python scripts/download_models.py
```

`requirements-gpu.txt` includes PyTorch (CUDA 12.8), `deep-sort-realtime`, and **OpenAI CLIP** from GitHub (needed for default `clip_RN50` embedder).

Optional: `pip install -r requirements-dev.txt` for pytest.

Verify GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Docker Deployment

For production on a Linux server — or local testing via Docker Desktop — see **[DEPLOY.md](DEPLOY.md)**.

Quick start:

```bash
docker compose build
docker compose run --rm verticalframe \
  /data/input/video.mp4 \
  --output /data/output/video-vertical.mp4
```

Place videos in `data/input/`; outputs go to `data/output/`.

---

## 🎬 Usage

### 1. Production Mode (Standard)
Produces a clean **9:16 Vertical Video** ready for social media.
```bash
python auto_reframe.py input.mp4 --output final_result.mp4
```

### 2. Debug/Director Mode
Produces a **Side-by-Side Video** (Original 16:9 + Vertical 9:16) with AI visualization overlays. Useful for checking why the AI made specific decisions.
```bash
python auto_reframe.py input.mp4 --debug-view --output debug_result.mp4
```

### 3. Saliency Only (B-Roll Mode)
Ignores people and focuses on "interesting things" (high contrast/motion).
```bash
python auto_reframe.py input.mp4 --saliency-only
```

---

## ⚙️ Configuration (`config.json`)

Fine-tune the camera personality:

```json
{
  "scanner": {
    "face_detector": "yolo",
    "yolo_face_model": "yolov8m-face",
    "batch_size": 128
  },
  "tracking": {
    "tracker_type": "deepsort",
    "deepsort": {
      "embedder": "clip_RN50"
    },
    "smooth_factor": 0.1,
    "easing_type": "sine_in_out"
  },
  "camera_control": {
    "dead_zone_percent": 0.10,
    "transition_mode": "smart"
  }
}
```

Use `"embedder": "mobilenet"` to skip CLIP (faster, slightly weaker tracking in crowds).

---

## 📂 Project Structure

*   `auto_reframe.py`: The Commander.
*   `modules/pipeline/scanner.py`: Detection & tracking pass (YOLO, DeepSORT, saliency).
*   `modules/pipeline/analyzer.py`: Camera path generation.
*   `modules/pipeline/renderer.py`: The **High-Fidelity Renderer** (FFmpeg Pipe Logic).
*   `modules/core/director.py`: Focus priority (Face > Body > Saliency).
*   `Dockerfile`, `docker-compose.yml`, `DEPLOY.md`: Server deployment.

---
*Powered by Advanced Agentic Coding - Google Deepmind*
