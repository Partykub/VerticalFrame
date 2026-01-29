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
                    
                    if current_frame_data:
                        for track in current_frame_data.get('tracks', []):
                            dx, dy, dw, dh = track['bbox']
                            cls_id = track['class_id']
                            color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)
                            cv2.rectangle(debug_frame, (dx, dy), (dx + dw, dy + dh), color, 2)
                    
                    # Draw Crop Box on Original
                    cv2.rectangle(debug_frame, (x1, 0), (x2, height), (255, 0, 255), 4)

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
