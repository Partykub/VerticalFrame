import argparse
import os
import sys

# Ensure modules are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.detection.saliency_pipeline import run_saliency_pipeline

import json

def load_config(config_path="config.json"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {config_path} not found. Using internal defaults.")
        return {}

def main():
    # Load Config
    config = load_config()
    tracking_cfg = config.get("tracking", {})
    pipeline_cfg = config.get("pipeline", {})

    parser = argparse.ArgumentParser(description="Auto-Reframe: Detection Pipeline Test")
    parser.add_argument("video_path", type=str, help="Path to input video file")
    
    # Use Config Defaults
    default_pipeline = pipeline_cfg.get("default", "yolo")
    default_smooth = tracking_cfg.get("smooth_factor", 0.1)
    default_sharpness = tracking_cfg.get("min_sharpness", 0)

    parser.add_argument("--mode", type=str, default="offline", choices=["realtime", "offline"], help="Processing mode: 'realtime' (fast) or 'offline' (best quality)")
    parser.add_argument("--pipeline", type=str, default=default_pipeline, choices=["yolo", "mediapipe", "saliency", "tracking"], help=f"Choose pipeline (default: {default_pipeline})")
    parser.add_argument("--algo", type=str, default="spectral", choices=["spectral", "fine_grained"], help="Saliency algorithm")
    parser.add_argument("--sharpness", type=float, default=default_sharpness, help=f"Minimum sharpness threshold (default: {default_sharpness})")
    parser.add_argument("--smooth", type=float, default=default_smooth, help=f"Smoothing factor (default: {default_smooth})")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video")
    parser.add_argument("--saliency-only", action="store_true", help="Run ONLY Saliency detection (Skip Face/Body)")
    
    args = parser.parse_args()

    if args.saliency_only:
        print("⚠️ Mode: SALIENCY ONLY (Skipping Face/Body Detection)")
        if 'scanner' not in config:
            config['scanner'] = {}
        config['scanner']['saliency_only'] = True
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
        return

    # --- OFFLINE MODE (Two-Pass System) ---
    if args.mode == "offline":
        if args.pipeline != "tracking":
            print("Error: Offline mode currently only supports 'tracking' pipeline.")
            return
            
        from modules.pipeline.scanner import VideoScanner
        from modules.pipeline.analyzer import VideoAnalyzer
        from modules.pipeline.renderer import VideoRenderer
        
        # Paths
        json_track_data = "temp_tracking_data.json"
        json_camera_path = "temp_camera_path.json"
        output_video = args.output if args.output else "output_offline_final.mp4"
        
        # 1. SCAN
        scanner = VideoScanner(config)
        scanner.scan(args.video_path, json_track_data)
        
        # 2. ANALYZE
        analyzer = VideoAnalyzer(config)
        analyzer.analyze(json_track_data, json_camera_path)
        
        # 3. RENDER
        renderer = VideoRenderer(config)
        renderer.render(args.video_path, json_camera_path, output_video, tracking_json_path=json_track_data)
        
        # 4. MERGE AUDIO (Robust Cross-Platform Logic)
        print("-" * 30)
        print(f"🎬 Merging audio from source: {args.video_path}...")
        
        import subprocess
        import shutil
        
        # Determine FFmpeg command
        ffmpeg_cmd = []
        
        if os.name == 'nt':
            # --- WINDOWS LOGIC ---
            # 1. Try Native Windows FFmpeg
            if shutil.which("ffmpeg.exe") or shutil.which("ffmpeg"):
                ffmpeg_cmd = ["ffmpeg"]
            # 2. Fallback to WSL (The fix for your case)
            elif shutil.which("wsl"):
                print("ℹ️  Windows Python detected: Using FFmpeg via WSL...")
                ffmpeg_cmd = ["wsl", "ffmpeg"]
        else:
            # --- LINUX / MAC LOGIC ---
            if shutil.which("ffmpeg"):
                ffmpeg_cmd = ["ffmpeg"]
            elif os.path.exists("/usr/bin/ffmpeg"):
                 ffmpeg_cmd = ["/usr/bin/ffmpeg"]

        if not ffmpeg_cmd:
            print("❌ Warning: FFmpeg not found. Video will be mute.")
            return

        # Define Temp and Final Paths
        # We need to rename the current mute output to temp, then merge to final
        temp_no_audio = output_video + ".temp_mute.mp4"
        
        try:
            # Rename rendered video to temp
            if os.path.exists(output_video):
                if os.path.exists(temp_no_audio):
                    os.remove(temp_no_audio)
                os.rename(output_video, temp_no_audio)
            
            # Construct Command
            cmd = ffmpeg_cmd + [
                "-y",
                "-i", temp_no_audio,     # Video Source
                "-i", args.video_path,   # Audio Source
                "-c:v", "copy",          # Copy Video Stream (No re-encode)
                "-c:a", "aac",           # Encode Audio to AAC
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",             # Stop when shortest stream ends
                output_video             # Final Result
            ]
            
            # Run
            subprocess.run(cmd, check=True)
            print(f"✅ Audio Merge Complete: {output_video}")
            
            # Cleanup
            if os.path.exists(temp_no_audio):
                os.remove(temp_no_audio)
                
        except Exception as e:
            print(f"❌ Failed to merge audio: {e}")
            # Restore original file if merge failed
            if os.path.exists(temp_no_audio) and not os.path.exists(output_video):
                os.rename(temp_no_audio, output_video)
        
        return

    # --- REALTIME MODE ---
    if args.pipeline == "yolo":
        from modules.detection.yolov8_pipeline import run_yolov8_pipeline
        run_yolov8_pipeline(args.video_path, args.output)
    elif args.pipeline == "mediapipe":
        from modules.detection.mediapipe_pipeline import run_mediapipe_pipeline
        run_mediapipe_pipeline(args.video_path, args.output)
    elif args.pipeline == "saliency":
        run_saliency_pipeline(args.video_path, args.output, algorithm=args.algo)
    elif args.pipeline == "tracking":
        from modules.tracking.tracking_pipeline import TrackingPipeline
        pipeline = TrackingPipeline(min_sharpness_threshold=args.sharpness, smooth_factor=args.smooth)
        pipeline.run(args.video_path, args.output)

if __name__ == "__main__":
    main()
