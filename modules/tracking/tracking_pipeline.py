import cv2
import time
import sys
from modules.detection.mediapipe_detector import MediaPipeDetector
from modules.detection.yolov8_detector import YOLOv8Detector
from modules.detection.saliency_detector import SaliencyDetector
from modules.tracking.tracker import ByteTracker
from modules.core.director import Director
import numpy as np

class TrackingPipeline:
    def __init__(self, min_sharpness_threshold=0, smooth_factor=0.1):
        # Lower confidence to 0.3 to let ByteTrack handle low-confidence matches
        self.detector = MediaPipeDetector(min_sharpness_threshold=min_sharpness_threshold, min_detection_confidence=0.3)
        self.tracker = ByteTracker(track_thresh=0.25, track_buffer=30)
        
        self.face_detector = self.detector # Alias
        self.body_detector = YOLOv8Detector(model_path="yolov8n.pt") 
        self.saliency_detector = SaliencyDetector(algorithm="spectral")
        self.director = Director()
        
        # Camera State
        self.smooth_factor = smooth_factor
        self.camera_x = None # Stores current 'virtual camera' center X position
        
        self.frame_count = 0
        self.start_time = 0

    def run(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Calculate Crop Dimensions (9:16)
        crop_h = height
        crop_w = int(crop_h * (9/16))
        
        # New Output Width = Original + Crop
        out_width = width + crop_w
        
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, height))

        self.start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Keep a clean copy for the actual crop result
            clean_frame = frame.copy()

            self.frame_count += 1
            if self.frame_count % 10 == 0:
                 print(f"Processing frame {self.frame_count}")

            # --- Detection Stage ---
            all_detections = []

            # 1. Face Detection
            face_results = self.face_detector.detect(frame)
            for det in face_results:
                det['class_id'] = 0 
                all_detections.append(det)

            # 2. Body Detection
            target_classes = [0, 2, 3, 15, 16, 63, 67]
            yolo_results = self.body_detector.detect(frame, classes=target_classes)
            for det in yolo_results:
                if det.get('class_id') == 0:
                    det['class_id'] = 1
                else:
                    det['class_id'] = 2
                all_detections.append(det)

            # 3. Saliency Detection (Compute Point)
            saliency_map = self.saliency_detector.detect(frame)
            saliency_point = None
            if saliency_map is not None:
                # Mask Logo (Top-Right 25%)
                h_map, w_map = saliency_map.shape
                mask_w = int(w_map * 0.25)
                mask_h = int(h_map * 0.25)
                cv2.rectangle(saliency_map, (w_map - mask_w, 0), (w_map, mask_h), 0, -1)
                
                # Find Max Point
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(saliency_map)
                saliency_point = maxLoc # (x, y)

            # --- Tracking Stage ---
            tracked_objects = self.tracker.update(all_detections)

            # --- Director Stage (Decision Making) ---
            target_point, decision_reason = self.director.select_target(
                tracked_objects, saliency_point, width, height
            )
            
            # --- Camera Smoothing Logic (Cinematic Feel) ---
            raw_target_x, _ = target_point
            
            if self.camera_x is None:
                self.camera_x = raw_target_x # Initialize at first target
            else:
                # Exponential Moving Average (EMA)
                # New = Old * (1-alpha) + Target * alpha
                self.camera_x = (self.camera_x * (1 - self.smooth_factor)) + (raw_target_x * self.smooth_factor)
                
            # Use smoothed X for cropping
            tx = int(self.camera_x) # Integer coordinate
            
            # Calculate Top-Left Corner (Centered on Target X)
            x1 = int(tx - (crop_w / 2))
            y1 = 0 # Full Height
            
            # Clamp to Frame Boundaries
            if x1 < 0: x1 = 0
            if x1 + crop_w > width: x1 = width - crop_w
            
            x2 = x1 + crop_w
            y2 = y1 + crop_h
            
            # Extract Clean Crop
            crop_view = clean_frame[y1:y2, x1:x2]

            # --- Visualization Stage (Draw on 'frame') ---
            # 1. Draw Tracked Objects
            for obj in tracked_objects:
                x, y, w, h = obj['bbox']
                cls_id = obj.get('class_id', 0)
                
                if cls_id == 0: color = (0, 255, 0) # Green (Face)
                elif cls_id == 1: color = (255, 0, 0) # Blue (Body)
                else: color = (0, 255, 255) # Yellow (Obj)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # 2. Draw Saliency Point
            if saliency_point:
                sx, sy = saliency_point
                cv2.drawMarker(frame, (sx, sy), (128, 128, 128), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            # 3. Draw Crop Window (9:16)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 4)
            cv2.line(frame, (x1 + crop_w//2, y1), (x1 + crop_w//2, y2), (255, 0, 255), 1)
            
            # 4. Draw Director Decision Text
            cv2.putText(frame, f"TARGET: {decision_reason}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)

            # --- Combine Views (Side-by-Side) ---
            # Ensure heights match (should be guaranteed by full-height crop)
            if crop_view.shape[0] != frame.shape[0]:
                crop_view = cv2.resize(crop_view, (crop_w, height))
                
            combined_view = np.hstack((frame, crop_view))

            if out:
                out.write(combined_view)

        cap.release()
        if out:
            out.release()
        
        end_time = time.time()
        print(f"Tracking finished. Processed {self.frame_count} frames in {end_time - self.start_time:.2f} seconds.")

def run(video_path, output_path, min_sharpness_threshold=0):
    pipeline = TrackingPipeline(min_sharpness_threshold=min_sharpness_threshold)
    pipeline.process_video(video_path, output_path)
