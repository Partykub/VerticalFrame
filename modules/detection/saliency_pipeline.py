import cv2
import time
import numpy as np
from .saliency_detector import SaliencyDetector

def run_saliency_pipeline(video_path, output_path, algorithm="spectral"):
    """
    Run the standalone Saliency detection pipeline.
    
    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to save the output video.
        algorithm (str): 'spectral' or 'fine_grained'
    """
    # 1. Input: Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 2. Initialize Saliency Detector
    detector = SaliencyDetector(algorithm=algorithm)

    # Output Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {video_path} with {algorithm} algorithm...")
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 3. Detection (Compute Saliency Map)
        saliency_map = detector.detect(frame)
        
        if saliency_map is not None:
            # 4. Visualization

            # --- Mask out Channel Logo (Top Right Corner) ---
            # Assume logo takes up ~25% of width and ~25% of height (Increased from 15%)
            h, w = saliency_map.shape
            logo_w = int(w * 0.13)
            logo_h = int(h * 0.18)
            # Set top-right corner to 0 (Black) so it's ignored
            saliency_map[0:logo_h, w-logo_w:w] = 0
            # -----------------------------------------------

            # Convert grayscale saliency map to color heatmap for background (optional, made fainter)
            heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
            blended_frame = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0) # Fainter heatmap

            # Draw a rectangle to show the masked area (for debugging)
            cv2.rectangle(blended_frame, (w-logo_w, 0), (w, logo_h), (0, 0, 0), 2)
            cv2.putText(blended_frame, "Ignored Area", (w-logo_w+10, logo_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Find the most salient point (Max Location)
            minVar, maxVal, minLoc, maxLoc = cv2.minMaxLoc(saliency_map)
            
            # Draw a crosshair/circle at the most salient point
            if maxVal > 50: # Only draw if there's significant saliency
                cv2.circle(blended_frame, maxLoc, 20, (0, 0, 255), 3) # Red Circle
                cv2.line(blended_frame, (maxLoc[0]-30, maxLoc[1]), (maxLoc[0]+30, maxLoc[1]), (0, 0, 255), 2)
                cv2.line(blended_frame, (maxLoc[0], maxLoc[1]-30), (maxLoc[0], maxLoc[1]+30), (0, 0, 255), 2)
                
                # Label coordinates
                cv2.putText(blended_frame, f"Max: {maxLoc}", (maxLoc[0]+25, maxLoc[1]-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Put text label
            cv2.putText(blended_frame, f"Saliency: {algorithm}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            out.write(blended_frame)
        else:
            # If failed, write original frame
            out.write(frame)
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames.")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Done! Processed {frame_count} frames in {duration:.2f} seconds.")
    print(f"Output saved to: {output_path}")
