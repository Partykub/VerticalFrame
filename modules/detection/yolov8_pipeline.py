import cv2
import time
from .yolov8_detector import YOLOv8Detector

def run_yolov8_pipeline(video_path, output_path):
    """
    Run the standalone YOLOv8 detection pipeline.
    
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
    
    # 2. Initialize YOLO Detector
    print("Initializing YOLOv8 Detector with Sharpness Filter...")
    # You can change model_path to 'yolov8s.pt' or 'yolov8m.pt' for better accuracy but slower speed
    detector = YOLOv8Detector(model_path="yolov8n.pt", conf_threshold=0.7, min_sharpness_threshold=15)

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
        
        # 3. Detection (Whitelist Important Classes)
        # 0:person, 2:car, 3:motorcycle, 15:cat, 16:dog, 63:laptop, 67:cell phone
        target_classes = [0, 2, 3, 15, 16, 63, 67]
        
        # Force Convert BGR -> RGB for Inference (In case Model expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect(frame_rgb, classes=target_classes)
        
        # Get class names from the model
        class_names = detector.model.names

        # 4. Visualization
        for det in detections:
            x, y, w, h = det['bbox']
            score = det['score']
            sharpness = det.get('sharpness', 0)
            class_id = det.get('class_id', 0)
            class_name = class_names.get(class_id, str(class_id))
            
            # Draw rectangle (Red for Body)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Draw label
            label = f"{class_name}: {score:.2f} | S:{sharpness:.0f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Write to output file
        out.write(frame)
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            # Count objects by class
            counts = {}
            for det in detections:
                cls_id = det.get('class_id')
                name = class_names.get(cls_id, 'unknown')
                counts[name] = counts.get(name, 0) + 1
            
            # Format log string
            summary = ", ".join([f"{count} {name}" for name, count in counts.items()])
            if not summary:
                summary = "None"
            print(f"Frame {frame_count}: Found [{summary}]")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Done! Processed {frame_count} frames in {duration:.2f} seconds.")
    print(f"Output saved to: {output_path}")
