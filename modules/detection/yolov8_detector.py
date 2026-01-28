from ultralytics import YOLO
import numpy as np
import torch
import cv2

class YOLOv8Detector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5, min_sharpness_threshold=0):
        """
        Initialize YOLOv8 Detector.
        """
        # Auto-select Device
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Loading YOLOv8 model: {model_path} on {self.device}...")
        
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.min_sharpness_threshold = min_sharpness_threshold

    def calculate_sharpness(self, image_roi):
        """
        Compute the Variance of Laplacian as a measure of sharpness.
        Higher value = Sharper image.
        """
        if image_roi is None or image_roi.shape[0] == 0 or image_roi.shape[1] == 0:
            return 0
        gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def detect(self, frame, classes=[0]):
        """
        Detect objects in a single frame.
        """
        if frame is None:
            return []
        return self.detect_batch([frame], classes=classes)[0]

    def detect_batch(self, frames, classes=[0]):
        """
        Detect objects in a batch of frames (List[np.array]).
        Returns List[List[Dict]].
        """
        batch_results = []
        if not frames:
            return batch_results

        # Run batch inference
        # Ultralytics accepts list of numpy arrays
        results_list = self.model(frames, verbose=False, classes=classes, conf=self.conf_threshold, iou=0.5, device=self.device)
        
        for i, result in enumerate(results_list):
            frame_results = []
            frame_h, frame_w = frames[i].shape[:2]
            
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
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
                    
                    roi = frames[i][roi_y1:roi_y2, roi_x1:roi_x2]
                    sharpness = self.calculate_sharpness(roi)
                    
                    if sharpness < self.min_sharpness_threshold:
                        continue
                
                frame_results.append({
                    "bbox": [x1, y1, w, h],
                    "score": score,
                    "sharpness": sharpness,
                    "type": "body" if cls == 0 else "object",
                    "class_id": cls
                })
            batch_results.append(frame_results)
            
        return batch_results
