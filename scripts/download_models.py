#!/usr/bin/env python3
"""Pre-download ML models for Docker build or manual setup."""

import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# Source: https://github.com/lindevs/yolov8-face
FACE_MODEL_URLS = {
    "yolov8n-face": "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt",
    "yolov8s-face": "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8s-face-lindevs.pt",
    "yolov8m-face": "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8m-face-lindevs.pt",
    "yolov8l-face": "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8l-face-lindevs.pt",
    "yolov8x-face": "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8x-face-lindevs.pt",
}

BODY_MODEL = "yolov8m.pt"


def load_config(path="config.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def download_file(url, dest):
    print(f"Downloading {dest} from {url}...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved {dest}")


def download_body_model():
    if os.path.exists(BODY_MODEL):
        print(f"{BODY_MODEL} already exists, skipping download.")
        return

    print(f"Downloading {BODY_MODEL} via ultralytics...")
    from ultralytics import YOLO

    YOLO(BODY_MODEL)
    if not os.path.exists(BODY_MODEL):
        raise RuntimeError(f"Failed to download {BODY_MODEL}")


def download_face_model(model_name):
    model_path = f"{model_name}.pt"
    if os.path.exists(model_path):
        print(f"{model_path} already exists, skipping download.")
        return

    url = FACE_MODEL_URLS.get(model_name)
    if not url:
        raise ValueError(f"No download URL for face model: {model_name}")

    download_file(url, model_path)


def warmup_deepsort(embedder):
    import numpy as np

    from modules.tracking.deepsort_tracker import DeepSortTracker

    print(f"Warming up DeepSORT embedder: {embedder}...")
    tracker = DeepSortTracker(embedder=embedder, embedder_gpu=False)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    tracker.update([], frame)
    print(f"DeepSORT embedder '{embedder}' ready.")


def main():
    config = load_config()
    scanner_cfg = config.get("scanner", {})
    face_model = scanner_cfg.get("yolo_face_model", "yolov8m-face")

    download_body_model()
    download_face_model(face_model)

    tracking_cfg = config.get("tracking", {})
    if tracking_cfg.get("tracker_type") == "deepsort":
        embedder = tracking_cfg.get("deepsort", {}).get("embedder", "mobilenet")
        warmup_deepsort(embedder)

    print("All models downloaded and warmed up.")


if __name__ == "__main__":
    main()
