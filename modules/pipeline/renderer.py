import cv2
import json
import numpy as np
from tqdm import tqdm

class VideoRenderer:
    def __init__(self, config):
        self.config = config

    def render(self, video_path, path_json_path, output_video_path, tracking_json_path=None, debug_mode=False):
        """
        Renders final video using generated camera path + overlays tracking data.
        Uses threaded IO for better performance.
        debug_mode: If True, outputs Side-by-Side view with debug overlays.
                    If False, outputs CLEAN Vertical 9:16 video.
        """
        print(f"Loading camera path from {path_json_path}...")
        with open(path_json_path, 'r') as f:
            data = json.load(f)
            
        camera_path = data['path']
        debug_info = data.get('debug_info', []) # Load reasoning info
        track_id_history = data.get('track_id_history', []) # Load focused Track IDs (decision)
        actual_focused_history = data.get('actual_focused_history', []) # ACTUAL bbox used for camera
        
        # Load config for threshold visualization
        cam_ctrl = self.config.get("camera_control", {})
        transition_mode = cam_ctrl.get("transition_mode", "smooth")
        smart_cut_threshold_pct = cam_ctrl.get("smart_cut_threshold_percent", 0.4)
        
        # Load Tracking Data
        tracking_frames = {}
        if tracking_json_path and debug_mode:
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
        
        # --- Output Dimensions ---
        if debug_mode:
            # Side-by-Side: Original (16:9) + Crop (9:16)
            out_width = width + crop_w
        else:
            # Clean Vertical (9:16)
            out_width = crop_w
        
        # FIX: libx264 requires dimensions to be divisible by 2
        if out_width % 2 != 0:
            out_width += 1
        
        import subprocess
        import shutil
        import os

        # --- FFmpeg Setup (High Quality / Visually Lossless) ---
        print(f"Initializing High-Quality FFmpeg Pipe ({'DEBUG SIDE-Z-SIDE' if debug_mode else 'CLEAN VERTICAL'})...")
        
        # Determine FFmpeg Executable List (Robust Cross-Platform)
        ffmpeg_cmd_list = []
        
        # ... (FFmpeg detection logic remains same) ...
        if os.name == 'nt':
            if shutil.which("ffmpeg.exe") or shutil.which("ffmpeg"): # No simple 'ffmpeg' on pure cmd sometimes
                 ffmpeg_cmd_list = ["ffmpeg"]
            elif shutil.which("wsl"):
                ffmpeg_cmd_list = ["wsl", "ffmpeg"]
            else:
                 print("Error: FFmpeg not found.")
                 return
        else:
             if shutil.which("ffmpeg"): ffmpeg_cmd_list = ["ffmpeg"]
             elif os.path.exists("/usr/bin/ffmpeg"): ffmpeg_cmd_list = ["/usr/bin/ffmpeg"]

        # CRF 18 = Visually Lossless
        cmd = ffmpeg_cmd_list + [
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{out_width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'slow', 
            '-crf', '18', 
            '-pix_fmt', 'yuv420p',
            output_video_path
        ]
        
        try:
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except Exception as e:
             print(f"❌ Failed to start FFmpeg process: {e}")
             return

        # ... (Threading setup remains same) ...
        # --- Threading Setup ---
        import threading
        import queue
        raw_queue = queue.Queue(maxsize=128)
        processed_queue = queue.Queue(maxsize=128)
        
        def reader_worker():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                raw_queue.put(frame)
            raw_queue.put(None)
            cap.release()

        def writer_worker():
            while True:
                frame = processed_queue.get()
                if frame is None:
                    processed_queue.task_done()
                    break
                try: process.stdin.write(frame.tobytes())
                except: break
                processed_queue.task_done()
            process.stdin.close()
            process.wait()

        t_read = threading.Thread(target=reader_worker, daemon=True)
        t_write = threading.Thread(target=writer_worker, daemon=True)
        t_read.start()
        t_write.start()
        
        print(f"Rendering Video to {output_video_path}...")
        pbar = tqdm(total=total_frames, unit="frame")
        
        frame_idx = 0
        
        while True:
            # 1. Get Raw Frame
            frame = raw_queue.get()
            if frame is None:
                processed_queue.put(None)
                break
                
            if frame_idx < len(camera_path):
                cam_x = camera_path[frame_idx]
                
                # Crop Logic (Always needed)
                x1 = int(cam_x - (crop_w / 2))
                if x1 < 0: x1 = 0
                if x1 + crop_w > width: x1 = width - crop_w
                x2 = x1 + crop_w
                
                # Clean Crop
                crop_view = frame[0:height, x1:x2]
                
                if debug_mode:
                    # --- DEBUG MODE: DRAWING & STACKING ---
                    debug_frame = frame.copy()
                    
                    # Draw Overlays
                    current_frame_data = tracking_frames.get(frame_idx + 1)
                    current_reason = debug_info[frame_idx] if frame_idx < len(debug_info) else ""
                    
                    # Get ACTUAL focused bbox info (used for camera position)
                    actual_focused_info = actual_focused_history[frame_idx] if frame_idx < len(actual_focused_history) else None
                    focused_track_id = actual_focused_info['track_id'] if actual_focused_info else None
                    focused_class_id = actual_focused_info['class_id'] if actual_focused_info else None
                    
                    if current_frame_data:
                        for track in current_frame_data.get('tracks', []):
                            # Skip Ghost Tracks (missed > 0)
                            if track.get('missed', 0) > 0:
                                continue

                            dx, dy, dw, dh = track['bbox']
                            cls_id = track['class_id']
                            track_id = track.get('id', '?')
                            conf = track.get('conf', 0.0) # Get confidence score
                            sharp = track.get('sharpness', 0.0) # Get sharpness score
                            
                            # Check if this is the ACTUAL focused track (used for camera position)
                            is_focused = (track_id == focused_track_id and cls_id == focused_class_id)
                            
                            # Color: Green for Face, Blue for Body
                            base_color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)
                            
                            # If focused: Use bright cyan/yellow, thicker line
                            if is_focused:
                                color = (0, 255, 255)  # Bright Cyan
                                thickness = 5
                                label_bg_color = (0, 200, 200)  # Darker cyan for label bg
                            else:
                                color = base_color
                                thickness = 2
                                label_bg_color = color
                            
                            # Draw bounding box
                            cv2.rectangle(debug_frame, (dx, dy), (dx + dw, dy + dh), color, thickness)
                            
                            # Draw Track ID label with Confidence and Sharpness
                            class_name = "Face" if cls_id == 0 else "Body"
                            label = f"{class_name} #{track_id} ({conf:.2f} S:{int(sharp)})"
                            if is_focused:
                                label = f">>> {class_name} #{track_id} ({conf:.2f} S:{int(sharp)}) [FOCUS] <<<"
                            
                            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            label_w, label_h = label_size
                            
                            # Background rectangle for label (above bbox)
                            label_y = max(dy - 5, label_h + 5)
                            cv2.rectangle(debug_frame, 
                                        (dx, label_y - label_h - 5), 
                                        (dx + label_w + 5, label_y + 2), 
                                        label_bg_color, -1)
                            
                            # Text label (white color, bold if focused)
                            text_thickness = 3 if is_focused else 2
                            cv2.putText(debug_frame, label, 
                                      (dx + 2, label_y - 2), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.6, (255, 255, 255), text_thickness)
                    
                    # Draw Crop Box on Original
                    cv2.rectangle(debug_frame, (x1, 0), (x2, height), (255, 0, 255), 4)
                    
                    # Draw Cut/Plan Threshold Lines or Predictive Status
                    if transition_mode == "predictive":
                        # Get predictive status for this frame
                        predictive_status = data.get('predictive_status', [])
                        plan_status = predictive_status[frame_idx] if frame_idx < len(predictive_status) else ""
                        
                        # Draw center camera position indicator
                        cv2.line(debug_frame, (int(cam_x), 0), 
                               (int(cam_x), height), 
                               (0, 165, 255), 2)  # Orange line = camera center
                        
                        # Color based on status
                        status_color = (0, 255, 0)  # Green for PLAN
                        if "PRE-PLAN" in plan_status:
                            status_color = (0, 165, 255)  # Orange
                        elif "HOLD" in plan_status:
                            status_color = (0, 255, 255)  # Yellow
                        elif "CUT" in plan_status:
                            status_color = (0, 0, 255)  # Red
                        
                        # Show mode and status
                        legend_y = 30
                        cv2.putText(debug_frame, f"Mode: PREDICTIVE", 
                                  (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (255, 255, 255), 2)
                        
                        if plan_status:
                            cv2.putText(debug_frame, plan_status, 
                                      (10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.6, status_color, 2)
                        
                        # Show focused Track ID
                        if actual_focused_info:
                            focused_track_id = actual_focused_info.get('track_id')
                            focused_class = actual_focused_info.get('class_id')
                            
                            if focused_track_id is not None and focused_class is not None:
                                class_name = "Face" if focused_class == 0 else "Body" if focused_class == 1 else "Obj"
                                focus_text = f"Focus: {class_name} ID:{focused_track_id}"
                            else:
                                focus_text = "Focus: Saliency/None"
                        else:
                            focus_text = "Focus: None"
                        
                        cv2.putText(debug_frame, focus_text, 
                                  (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.6, (255, 255, 255), 2)
                    
                    elif transition_mode in ["smart", "conversation"]:
                        threshold_px = int(width * smart_cut_threshold_pct)
                        
                        # Left threshold line (ถ้าเกินเส้นนี้ไปทางซ้าย = cut)
                        left_threshold = cam_x - threshold_px
                        if left_threshold > 0:
                            cv2.line(debug_frame, (int(left_threshold), 0), 
                                   (int(left_threshold), height), 
                                   (0, 255, 255), 2)  # Yellow line
                        
                        # Right threshold line (ถ้าเกินเส้นนี้ไปทางขวา = cut)
                        right_threshold = cam_x + threshold_px
                        if right_threshold < width:
                            cv2.line(debug_frame, (int(right_threshold), 0), 
                                   (int(right_threshold), height), 
                                   (0, 255, 255), 2)  # Yellow line
                        
                        # Draw center camera position indicator
                        cv2.line(debug_frame, (int(cam_x), 0), 
                               (int(cam_x), height), 
                               (0, 165, 255), 2)  # Orange line = camera center
                        
                        # Add legend text
                        legend_y = 30
                        cv2.putText(debug_frame, f"Mode: {transition_mode.upper()}", 
                                  (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (255, 255, 255), 2)
                        cv2.putText(debug_frame, f"Cut Threshold: {int(threshold_px)}px ({smart_cut_threshold_pct*100:.0f}%)", 
                                  (10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.6, (0, 255, 255), 2)
                        
                        # Show focused Track ID (ACTUAL bbox used for camera)
                        if actual_focused_info:
                            focused_track_id = actual_focused_info.get('track_id')
                            focused_class = actual_focused_info.get('class_id')
                            
                            if focused_track_id is not None and focused_class is not None:
                                class_name = "Face" if focused_class == 0 else "Body" if focused_class == 1 else "Obj"
                                focus_text = f"Focus: {class_name} ID:{focused_track_id}"
                            else:
                                # Saliency or other non-tracked target
                                focus_text = "Focus: Saliency/None"
                        else:
                            focus_text = "Focus: None (No Data)"
                        
                        cv2.putText(debug_frame, focus_text, 
                                  (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.6, (0, 255, 255), 2)  # Cyan color
                        
                        # Show decision Track ID (what Analyzer wanted to lock)
                        decision_id = track_id_history[frame_idx] if frame_idx < len(track_id_history) else None
                        if decision_id is not None:
                            decision_text = f"Decision: Track ID:{decision_id}"
                        else:
                            decision_text = "Decision: None"
                        
                        cv2.putText(debug_frame, decision_text, 
                                  (10, legend_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.6, (100, 200, 255), 2)  # Light orange
                        
                        # Current reason (from analyzer)
                        if current_reason:
                            cv2.putText(debug_frame, f"Reason: {current_reason}", 
                                      (10, legend_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (200, 200, 200), 2)  # Gray

                    # Resize crop if needed (should match height already)
                    if crop_view.shape[0] != height:
                         crop_view = cv2.resize(crop_view, (crop_w, height))

                    combined = np.hstack((debug_frame, crop_view))
                else:
                    # --- PRODUCTION MODE: CLEAN CROP ONLY ---
                    combined = crop_view

                # --- FIX SKEW/ALINGMENT ISSUES (Apply to both modes) ---
                current_w = combined.shape[1]
                if current_w != out_width:
                    pad_w = out_width - current_w
                    combined = cv2.copyMakeBorder(combined, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                
                processed_queue.put(combined)
            else:
                pass # End of path

            frame_idx += 1
            pbar.update(1)
            
        # ... (Cleanup remains same) ...
        t_read.join()
        t_write.join()
        pbar.close()
        
        # Integrity Check
        print("-" * 30)
        print(f"📊 Rendering Statistics (Final): {frame_idx}/{total_frames} frames")
        print("-" * 30)
        print("Rendering Complete.")
