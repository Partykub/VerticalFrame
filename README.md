# Auto-Reframe Pipeline

An intelligent, AI-powered automated video reframing tool designed to crop horizontal videos (16:9) into vertical (9:16) formats suitable for TikTok, Reels, and Shorts. It uses computer vision to track subjects, apply cinematic smoothing, and intelligently manage camera cuts.

## Key Features

*   **Multi-Stage Detection**: Hybrid pipeline using **YOLOv8** (Objects), **MediaPipe** (Face/Pose), and **Spectral Residual Saliency** (Attention).
*   **Smart Director AI**: Intelligently prioritizes targets: `Face > Body > Saliency`.
*   **Cinematic Camera Smoothing**:
    *   **7 Easing Types**: `linear`, `ease_in`, `ease_out`, `sine_in_out` (and more) for natural camera movement.
    *   **Smart Lock**: Predicts future movements to prevent camera jitter.
    *   **Dead Zone Stabilization**: Keeps the camera steady when subject movement is minimal.
*   **Audio Merging**: Automatically preserves original audio in the final output.
*   **Cross-Platform**: Works on **Windows**, **Linux**, **WSL**, and **macOS**.

## Installation

### Prerequisites
1.  **Python 3.8+**
2.  **FFmpeg** (Required for audio merging)
    *   *Ubuntu/WSL*: `sudo apt install ffmpeg`
    *   *Mac*: `brew install ffmpeg`
    *   *Windows*: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### Setup
1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/auto-reframe.git
    cd auto-reframe
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Basic Command
Run the pipeline in `offline` mode (recommended for best quality):

```bash
python auto_reframe.py input_video.mp4 --mode offline --output output.mp4
```

### Advanced Options
*   `--saliency-only`: Focus only on interesting areas (ignore people).
*   `--smooth`: Adjust smoothing factor (0.1 = smooth, 0.9 = fast).
*   `--sharpness`: Minimum sharpness threshold to ignore blurry frames.

## ⚙️ Configuration (`config.json`)

You can fine-tune the behavior in `config.json`. Key settings:

```json
{
  "tracking": {
    "easing_type": "sine_in_out",  // Options: ease_in, ease_out, linear...
    "smooth_factor": 0.1
  },
  "camera_control": {
    "dead_zone_percent": 0.05,     // Area where camera won't move
    "look_ahead_frames": 60        // How far to predict future movement
  },
  "saliency_control": {
    "ignore_border_percent": 0.15  // Ignore edges of the screen
  }
}
```

## Project Structure

*   `auto_reframe.py`: Main entry point.
*   `modules/pipeline/`: Core logic (Scanner, Analyzer, Renderer).
*   `modules/detection/`: AI Models (YOLO, MediaPipe).
*   `modules/core/`: Decision making logic (Director).

---
*Created for automated content repurposing workflows.*
