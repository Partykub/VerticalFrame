import cv2
import time
from .mediapipe_detector import MediaPipeDetector

def run_mediapipe_pipeline(video_path, output_path):
    """
    Run the standalone MediaPipe detection pipeline.
    
    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to save the output video.
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
    
    # 2. Initialize MediaPipe Detector
    print("Initializing MediaPipe Detector with Sharpness Filter...")
    detector = MediaPipeDetector(model_selection=1, min_detection_confidence=0.7, min_sharpness_threshold=10) # Threshold 100 is a good starting point

    # Output Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {video_path}...")
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 3. Detection
        detections = detector.detect(frame)
        
        # 4. Visualization (Draw Bounding Boxes)
        for det in detections:
            x, y, w, h = det['bbox']
            score = det['score']
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            sharpness = det.get('sharpness', 0)
            label = f"Face: {score:.2f} | Sharp: {sharpness:.0f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write to output file
        out.write(frame)
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames. Found {len(detections)} faces in current frame.")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Done! Processed {frame_count} frames in {duration:.2f} seconds.")
    print(f"Output saved to: {output_path}")
