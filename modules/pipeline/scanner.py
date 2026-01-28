import cv2
import json
import time
import os
import numpy as np
from tqdm import tqdm
from modules.detection.mediapipe_detector import MediaPipeDetector
from modules.detection.yolov8_detector import YOLOv8Detector
from modules.detection.saliency_detector import SaliencyDetector
from modules.tracking.tracker import ByteTracker

class VideoScanner:
    def __init__(self, config):
        self.config = config
        tracking_cfg = self.config.get("tracking", {})
        
        # Detector Initialization
        scanner_cfg = self.config.get("scanner", {})
        self.saliency_only = scanner_cfg.get("saliency_only", False)
        
        # Face: Use Pool for Parallel Processing
        import queue
        self.num_cpu_workers = 4 # Adjust based on CPU cores
        
        if not self.saliency_only:
            print(f"Initializing {self.num_cpu_workers} Face Detectors for Parallel CPU Processing...")
            self.face_detector_pool = queue.Queue()
            
            def create_detector():
                return MediaPipeDetector(
                    min_sharpness_threshold=tracking_cfg.get("min_sharpness", 0),
                    min_detection_confidence=tracking_cfg.get("detection_confidence", 0.3)
                )

            for _ in range(self.num_cpu_workers):
                self.face_detector_pool.put(create_detector())
            
            # Fallback ref
            self.face_detector = create_detector()
            
            # Body: Upgrade to YOLOv8m (Medium) for better separation
            print("Loading YOLOv8m (Medium) model...")
            self.body_detector = YOLOv8Detector(
                model_path="yolov8m.pt",
                min_sharpness_threshold=tracking_cfg.get("min_sharpness", 0)
            ) 
        else:
             print("Skipping Face/Body Model Loading (Saliency Only Mode)")
             self.face_detector_pool = None
             self.body_detector = None
        
        # Saliency
        self.saliency_detector = SaliencyDetector(algorithm="spectral")
        
        # Tracker
        self.tracker = ByteTracker(
            track_thresh=0.25, 
            track_buffer=tracking_cfg.get("track_buffer", 30)
        )

    def scan(self, video_path, output_json_path):
        """
        Runs full detection & tracking pass and saves results to JSON.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        tracking_data = {
            "meta": {
                "video_path": video_path,
                "width": width,
                "height": height,
                "total_frames": total_frames,
                "scan_time": time.ctime()
            },
            "frames": []
        }
        
        print(f"Starting Scan (Phase 1/3)... Frames: {total_frames}")
        
        scanner_cfg = self.config.get("scanner", {})
        BATCH_SIZE = scanner_cfg.get("batch_size", 16)
        print(f"Using Batch Size: {BATCH_SIZE}")
        
        # --- Multi-Threaded Video Reader Setup ---
        import threading
        import queue
        
        frame_queue = queue.Queue(maxsize=BATCH_SIZE * 4) # Buffer 4 batches ahead
        
        def read_frames():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_queue.put(frame)
            frame_queue.put(None) # Sentinel for EOF
            cap.release()
            
        reader_thread = threading.Thread(target=read_frames)
        reader_thread.daemon = True
        reader_thread.start()
        
        # --- Main Processing Loop ---
        frame_idx = 0
        pbar = tqdm(total=total_frames, unit="frame")
        
        frame_buffer = []
        
        # Init CPU Executor
        from concurrent.futures import ThreadPoolExecutor
        self.cpu_executor = ThreadPoolExecutor(max_workers=self.num_cpu_workers)
        
        while True:
            # Get frame from Queue
            frame = frame_queue.get()
            
            if frame is None:
                # EOF Sentinel
                if len(frame_buffer) > 0:
                    self.process_batch(frame_buffer, frame_idx, tracking_data, pbar)
                break
                
            frame_buffer.append(frame)
            
            if len(frame_buffer) >= BATCH_SIZE:
                self.process_batch(frame_buffer, frame_idx, tracking_data, pbar)
                frame_buffer = []

        reader_thread.join()
        self.cpu_executor.shutdown()
        pbar.close()
        
        # Save to JSON
        print(f"Saving data to {output_json_path}...")
        with open(output_json_path, 'w') as f:
            json.dump(tracking_data, f, indent=None) # No indent to save space
            
        print("Scan Complete.")

    def _process_cpu_task(self, frame):
        """Helper for parallel CPU processing"""
        # If saliency only, this shouldn't be called via executor map with this logic, 
        # But for safety/robustness:
        if self.saliency_only:
             return [], None 

        detector = self.face_detector_pool.get()
        try:
            face_results = detector.detect(frame)
           # 2. Saliency
            saliency_map = self.saliency_detector.detect(frame)
            saliency_point = None
            if saliency_map is not None:
                h_map, w_map = saliency_map.shape
                
                # Read Config for Border Mask (Same as Saliency Only Mode)
                sal_cfg = self.config.get("saliency_control", {})
                ignore_pct = sal_cfg.get("ignore_border_percent", 0.15)
                
                # Mask borders
                mask_w = int(w_map * ignore_pct)
                mask_h = int(h_map * ignore_pct)
                
                # Draw masks (Black out borders)
                cv2.rectangle(saliency_map, (0, 0), (w_map, mask_h), 0, -1) # Top
                cv2.rectangle(saliency_map, (0, h_map - mask_h), (w_map, h_map), 0, -1) # Bottom
                cv2.rectangle(saliency_map, (0, 0), (mask_w, h_map), 0, -1) # Left
                cv2.rectangle(saliency_map, (w_map - mask_w, 0), (w_map, h_map), 0, -1) # Right
                
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(saliency_map)
                saliency_point = maxLoc
                
            return face_results, saliency_point
        finally:
            self.face_detector_pool.put(detector)

    def process_batch(self, frames, current_global_idx, tracking_data, pbar):
        """
        Process a batch: YOLO (GPU Batch) + Face/Saliency (CPU Parallel)
        """
        yolo_batch_results = []
        cpu_results = []
        
        if not self.saliency_only:
            # 1. YOLO Batch Inference (GPU)
            target_classes = [0, 2, 3, 15, 16, 63, 67]
            yolo_batch_results = self.body_detector.detect_batch(frames, classes=target_classes)
            
            # 2. Parallel CPU Processing (Face + Saliency)
            # Use helper to run both Face and Saliency in parallel
            cpu_results = list(self.cpu_executor.map(self._process_cpu_task, frames))
        else:
            # Saliency Only Mode: Run ONLY Saliency (No Face Detector)
            for frame in frames:
                s_map = self.saliency_detector.detect(frame)
                
                # Apply Saliency Masking Logic
                s_point = None
                if s_map is not None:
                    h_map, w_map = s_map.shape
                    
                    # Read Config for Border Mask
                    sal_cfg = self.config.get("saliency_control", {})
                    ignore_pct = sal_cfg.get("ignore_border_percent", 0.15)
                    
                    # Mask borders
                    mask_w = int(w_map * ignore_pct)
                    mask_h = int(h_map * ignore_pct)
                    
                    # Draw masks
                    cv2.rectangle(s_map, (0, 0), (w_map, mask_h), 0, -1) # Top
                    cv2.rectangle(s_map, (0, h_map - mask_h), (w_map, h_map), 0, -1) # Bottom
                    cv2.rectangle(s_map, (0, 0), (mask_w, h_map), 0, -1) # Left
                    cv2.rectangle(s_map, (w_map - mask_w, 0), (w_map, h_map), 0, -1) # Right
                    
                    _, _, _, maxLoc = cv2.minMaxLoc(s_map)
                    s_point = maxLoc
                
                cpu_results.append(([], s_point)) # Empty face results
            
            # Fill dummy YOLO results
            yolo_batch_results = [[] for _ in frames]
        
        # 3. Aggregate & Track (Sequential)
        for i, frame in enumerate(frames):
            frame_seq_id = len(tracking_data["frames"]) + 1
            all_detections = []
            
            # Unpack CPU results
            face_results, saliency_point = cpu_results[i]
            
            # Face
            for det in face_results:
                det['class_id'] = 0 
                all_detections.append(det)
                
            # Body (from YOLO Batch)
            yolo_results = yolo_batch_results[i]
            for det in yolo_results:
                if det.get('class_id') == 0:
                    det['class_id'] = 1
                else:
                    det['class_id'] = 2
                all_detections.append(det)

            # Tracking
            tracked_objects = self.tracker.update(all_detections)
            
            # Serialize
            serialized_tracks = []
            for t in tracked_objects:
                bbox = [int(x) for x in t['bbox']]
                serialized_tracks.append({
                    "id": int(t['track_id']),
                    "class_id": int(t.get('class_id', -1)),
                    "bbox": bbox,
                    "conf": float(t.get('confidence', 0.0))
                })
            
            frame_data = {
                "frame_id": frame_seq_id,
                "tracks": serialized_tracks,
                "saliency_point": [int(x) for x in saliency_point] if saliency_point else None
            }
            
            tracking_data["frames"].append(frame_data)
            pbar.update(1)
