"""
YOLO Face Detector - ใช้ YOLO model ที่ train มาสำหรับ Face Detection
"""

from ultralytics import YOLO
import numpy as np
import torch
import cv2
import os

class YOLOFaceDetector:
    def __init__(self, model_name="yolov8n-face", conf_threshold=0.5, min_sharpness_threshold=0):
        """
        Initialize YOLO Face Detector.
        
        Args:
            model_name: "yolov8n-face", "yolov8s-face", "yolov8m-face"
            conf_threshold: Minimum confidence score
            min_sharpness_threshold: Minimum sharpness (0 to disable)
        """
        self.conf_threshold = conf_threshold
        self.min_sharpness_threshold = min_sharpness_threshold
        
        # Auto-select Device
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Model paths (will be downloaded if not exists)
        # Source: https://github.com/lindevs/yolov8-face
        model_urls = {
            "yolov8n-face": "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt",
            "yolov8s-face": "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8s-face-lindevs.pt",
            "yolov8m-face": "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8m-face-lindevs.pt",
            "yolov8l-face": "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8l-face-lindevs.pt",
            "yolov8x-face": "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8x-face-lindevs.pt"
        }
        
        model_path = f"{model_name}.pt"
        
        # Download model if not exists
        if not os.path.exists(model_path):
            url = model_urls.get(model_name)
            if url:
                print(f"Downloading {model_name} from lindevs/yolov8-face...")
                import urllib.request
                try:
                    urllib.request.urlretrieve(url, model_path)
                    print("Download complete.")
                except Exception as e:
                    raise RuntimeError(f"Failed to download {model_name}: {e}")
        
        print(f"Loading YOLO Face model: {model_path} on {self.device}...")
        self.model = YOLO(model_path)
    
    def calculate_sharpness(self, image_roi):
        """Compute the Variance of Laplacian as a measure of sharpness."""
        if image_roi is None or image_roi.shape[0] == 0 or image_roi.shape[1] == 0:
            return 0
        gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def detect(self, frame):
        """Detect faces in a single frame."""
        if frame is None:
            return []
        
        results = self.model(frame, verbose=False, conf=self.conf_threshold, device=self.device)
        
        detections = []
        frame_h, frame_w = frame.shape[:2]
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = float(box.conf[0].cpu().numpy())
                
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w = x2 - x1
                h = y2 - y1
                
                # Check sharpness if enabled
                sharpness = 0
                if self.min_sharpness_threshold > 0:
                    roi_x1 = max(0, x1)
                    roi_y1 = max(0, y1)
                    roi_x2 = min(frame_w, x2)
                    roi_y2 = min(frame_h, y2)
                    
                    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    sharpness = self.calculate_sharpness(roi)
                    
                    if sharpness < self.min_sharpness_threshold:
                        continue
                
                detections.append({
                    "bbox": [x1, y1, w, h],
                    "score": score,
                    "sharpness": sharpness,
                    "type": "face",
                    "class_id": 0  # Face
                })
        
        return detections
    
    def close(self):
        """Cleanup (YOLO doesn't need explicit cleanup)."""
        pass
