import cv2
import json
import numpy as np
from tqdm import tqdm

class VideoRenderer:
    def __init__(self, config):
        self.config = config

    def render(self, video_path, path_json_path, output_video_path, tracking_json_path=None):
        """
        Renders final video using generated camera path + overlays tracking data.
        Uses threaded IO for better performance.
        """
        print(f"Loading camera path from {path_json_path}...")
        with open(path_json_path, 'r') as f:
            data = json.load(f)
            
        camera_path = data['path']
        debug_info = data.get('debug_info', []) # Load reasoning info
        
        # Load Tracking Data
        tracking_frames = {}
        if tracking_json_path:
            print(f"Loading tracking data for visualization from {tracking_json_path}...")
            with open(tracking_json_path, 'r') as f:
                t_data = json.load(f)
                for fd in t_data['frames']:
                    tracking_frames[fd['frame_id']] = fd
        
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) # Use float fps for sync
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        crop_h = height
        crop_w = int(crop_h * (9/16))
        out_width = width + crop_w
        
        # Output writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, height))
        
        # --- Threading Setup ---
        import threading
        import queue
        
        # Limit queue size to prevent memory explosion
        raw_queue = queue.Queue(maxsize=128)
        processed_queue = queue.Queue(maxsize=128)
        
        def reader_worker():
            """Reads frames from video file"""
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                raw_queue.put(frame)
            raw_queue.put(None) # EOF
            cap.release()

        def writer_worker():
            """Writes processed frames to output file"""
            while True:
                frame = processed_queue.get()
                if frame is None:
                    processed_queue.task_done()
                    break
                out.write(frame)
                processed_queue.task_done()
            out.release()

        # Start IO Threads
        t_read = threading.Thread(target=reader_worker, daemon=True)
        t_write = threading.Thread(target=writer_worker, daemon=True)
        t_read.start()
        t_write.start()
        
        print(f"Rendering Video to {output_video_path} (Threaded)...")
        pbar = tqdm(total=total_frames, unit="frame")
        
        frame_idx = 0
        
        while True:
            # 1. Get Raw Frame
            frame = raw_queue.get()
            if frame is None:
                # End of Input
                processed_queue.put(None) # Signal Writer to stop
                break
                
            if frame_idx < len(camera_path):
                cam_x = camera_path[frame_idx]
                
                # --- PROCESSING (CPU Heavy) ---
                debug_frame = frame.copy()
                
                # Draw Overlays
                current_frame_data = tracking_frames.get(frame_idx + 1)
                
                # Get Decision Reason
                current_reason = debug_info[frame_idx] if frame_idx < len(debug_info) else ""
                
                if current_frame_data:
                    # Draw Tracks
                    for track in current_frame_data.get('tracks', []):
                        x, y, w, h = track['bbox']
                        cls_id = track['class_id']
                        track_id = track['id']
                        
                        color = (0, 255, 255)
                        label = f"Obj:{track_id}"
                        if cls_id == 0: 
                            color = (0, 255, 0); label = f"Face:{track_id}"
                        elif cls_id == 1: 
                            color = (255, 0, 0); label = f"Body:{track_id}"
                            
                        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(debug_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # Draw Saliency ONLY if selected
                    if "Saliency" in current_reason:
                        saliency_pt = current_frame_data.get('saliency_point')
                        if saliency_pt:
                            sx, sy = saliency_pt
                            # Draw visually distinct marker (Purple Crosshair)
                            cv2.circle(debug_frame, (sx, sy), 10, (255, 0, 255), 2)
                            cv2.line(debug_frame, (sx - 15, sy), (sx + 15, sy), (255, 0, 255), 2)
                            cv2.line(debug_frame, (sx, sy - 15), (sx, sy + 15), (255, 0, 255), 2)
                            cv2.putText(debug_frame, "SALIENCY", (sx + 15, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                # --- VISUALIZE IGNORED BORDERS (Saliency Only Mode) ---
                if self.config.get("scanner", {}).get("saliency_only", False):
                    overlay = debug_frame.copy()
                    h, w = debug_frame.shape[:2]
                    
                    # Read Config
                    sal_cfg = self.config.get("saliency_control", {})
                    ignore_pct = sal_cfg.get("ignore_border_percent", 0.15)
                    
                    mask_w = int(w * ignore_pct)
                    mask_h = int(h * ignore_pct)
                    
                    # Draw gray rectangles on borders
                    mask_color = (0, 0, 0) # Black
                    # Top
                    cv2.rectangle(overlay, (0, 0), (w, mask_h), mask_color, -1)
                    # Bottom
                    cv2.rectangle(overlay, (0, h - mask_h), (w, h), mask_color, -1)
                    # Left
                    cv2.rectangle(overlay, (0, 0), (mask_w, h), mask_color, -1)
                    # Right
                    cv2.rectangle(overlay, (w - mask_w, 0), (w, h), mask_color, -1)
                    
                    # Apply transparency (alpha = 0.5)
                    cv2.addWeighted(overlay, 0.5, debug_frame, 0.5, 0, debug_frame)
                    
                    # Add Label
                    cv2.putText(debug_frame, f"IGNORED REGION ({int(ignore_pct*100)}%)", (20, mask_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                # -----------------------------------------------------

                # Crop Logic
                x1 = int(cam_x - (crop_w / 2))
                if x1 < 0: x1 = 0
                if x1 + crop_w > width: x1 = width - crop_w
                x2 = x1 + crop_w
                
                crop_view = frame[0:height, x1:x2]
                
                # Draw Crop Box
                cv2.rectangle(debug_frame, (x1, 0), (x2, height), (255, 0, 255), 4)
                
                # Show Reason Text on top left
                reason_color = (0, 255, 255) # Yellow
                if "Locked" in current_reason: reason_color = (0, 255, 0) # Green for Locked
                elif "Saliency" in current_reason: reason_color = (255, 0, 255) # Purple
                
                cv2.putText(debug_frame, f"MODE: {current_reason}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, reason_color, 2)
                cv2.putText(debug_frame, "AUTO-REFRAME", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                
                # Stack
                if crop_view.shape[0] != height:
                     crop_view = cv2.resize(crop_view, (crop_w, height))
                
                combined = np.hstack((debug_frame, crop_view))
                
                # 2. Send to Writer
                processed_queue.put(combined)
            else:
                # If camera path shorter than video, just drop or handle gracefully
                pass

            frame_idx += 1
            pbar.update(1)
            
        # Wait for threads to finish
        t_read.join()
        t_write.join()
        pbar.close()
        print("Rendering Complete.")
