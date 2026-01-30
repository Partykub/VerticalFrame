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

*   **Multi-Stage AI**: Hybrid tracking using **YOLOv8** (People), **MediaPipe** (Faces), and **Spectral Residual Saliency** (Attention).
*   **Smart Director**: A decision engine that prioritizes `Face > Body > Saliency` based on stability and size.
*   **Cinematic Camera**:
    *   **Sine-In-Out Easing**: Organic start/stop camera movements (no robotic jerks).
    *   **Look-Ahead Logic**: The AI "sees the future" to prepare for subject movement before it happens.
    *   **Smart Lock**: Locks onto a specific "Actor ID" to prevent camera jumping in crowds.
*   **Vertical-First**: Default output is a clean, broadcast-ready 9:16 video.

---

## 📦 Installation

### Prerequisites
1.  **Python 3.8+**
2.  **FFmpeg** (CRITICAL: The core engine)
    *   *Ubuntu/WSL*: `sudo apt install ffmpeg`
    *   *Windows*: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### Setup
```bash
# 1. Install Dependencies
pip install -r requirements.txt
```

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
  "tracking": {
    "easing_type": "sine_in_out",   // Movement style: linear, ease_in, sine_in_out...
    "smooth_factor": 0.1,           // Lower = Smoother/Slower
    "min_sharpness": 50             // Ignore blurry frames
  },
  "camera_control": {
    "dead_zone_percent": 0.05,      // Sensitivity buffer
    "look_ahead_frames": 60         // Prediction window (~2 seconds)
  }
}
```

---

## 📂 Project Structure

*   `auto_reframe.py`: The Commander.
*   `modules/pipeline/renderer.py`: The **High-Fidelity Renderer** (FFmpeg Pipe Logic).
*   `modules/core/cameraman.py`: The **Camera Operator** (Smoothing & Easing Logic).
*   `modules/detection/`: The **Eyes** (YOLO/MediaPipe Wrappers).

---
*Powered by Advanced Agentic Coding - Google Deepmind*
