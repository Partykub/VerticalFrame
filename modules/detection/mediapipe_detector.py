import cv2
import mediapipe as mp
import numpy as np
import os

class MediaPipeDetector:
    def __init__(self, model_selection=1, min_detection_confidence=0.8, min_sharpness_threshold=0, model_type="short_range"):
        """
        Initialize MediaPipe Face Detection.
        model_selection: 0 for close range (Selfie), 1 for far range (Full body) - legacy parameter
        min_detection_confidence: Minimum confidence score [0.0, 1.0]
        min_sharpness_threshold: Minimum variance of Laplacian for blur detection (0 to disable, typical ~50-100)
        model_type: "short_range" (≤2m, selfie/webcam) or "full_range" (≤5m, far faces)
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_sharpness_threshold = min_sharpness_threshold
        self.model_type = model_type
        
        # MediaPipe 0.10.x uses tasks API instead of solutions
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        # NOTE: MediaPipe Tasks API currently only supports blaze_face_short_range
        # full_range model is not yet compatible with Tasks API (as of 2024)
        if model_type == "full_range":
            print("⚠️  Warning: 'full_range' model is NOT supported by MediaPipe Tasks API.")
            print("   Falling back to 'short_range'. For far faces, try lowering confidence threshold.")
            model_type = "short_range"
            self.model_type = model_type
        
        # Model URL (only short_range is supported)
        model_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        model_name = "blaze_face_short_range.tflite"
        
        if not os.path.exists(model_name):
            print(f"Downloading {model_name}...")
            import urllib.request
            try:
                urllib.request.urlretrieve(model_url, model_name)
                print("Download complete.")
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}")
        
        # Create FaceDetector with new API
        base_options = python.BaseOptions(model_asset_path=model_name)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence
        )
        self.face_detection = vision.FaceDetector.create_from_options(options)

    def close(self):
        """Explicitly close the MediaPipe FaceDetector resource."""
        if hasattr(self, 'face_detection') and self.face_detection:
            self.face_detection.close()
            self.face_detection = None

    def calculate_sharpness(self, image_roi):
        """
        Compute the Variance of Laplacian as a measure of sharpness.
        Higher value = Sharper image.
        """
        if image_roi is None or image_roi.shape[0] == 0 or image_roi.shape[1] == 0:
            return 0
        gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def detect(self, frame):
        """
        Detect faces in the frame.
        Returns a list of bounding boxes [x, y, w, h], scores, and sharpness.
        """
        results = []
        if frame is None:
            return results

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = frame.shape
        
        # Convert to MediaPipe Image format (0.10.x)
        from mediapipe import Image, ImageFormat
        mp_image = Image(image_format=ImageFormat.SRGB, data=image_rgb)
        
        # Process the image and find faces
        detection_results = self.face_detection.detect(mp_image)
        
        if detection_results.detections:
            for detection in detection_results.detections:
                # Get bounding box
                bbox = detection.bounding_box
                score = detection.categories[0].score
                
                # Manual filtering to ensure strict confidence threshold
                if score < self.min_detection_confidence:
                    continue
                
                # Convert to [x, y, w, h] format
                x = bbox.origin_x
                y = bbox.origin_y
                width = bbox.width
                height = bbox.height

                # Clamp coordinates to image boundaries
                x = max(0, x)
                y = max(0, y)
                width = min(width, w_img - x)
                height = min(height, h_img - y)
                
                sharpness = 0
                if self.min_sharpness_threshold > 0:
                    face_roi = frame[y:y+height, x:x+width]
                    sharpness = self.calculate_sharpness(face_roi)
                    
                    if sharpness < self.min_sharpness_threshold:
                        continue
                
                results.append({
                    "bbox": [x, y, width, height],
                    "score": score,
                    "sharpness": sharpness,
                    "type": "face"
                })
        
        return results
