import cv2
import mediapipe as mp
import numpy as np

class MediaPipeDetector:
    def __init__(self, model_selection=1, min_detection_confidence=0.8, min_sharpness_threshold=0):
        """
        Initialize MediaPipe Face Detection.
        model_selection: 0 for close range (Selfie), 1 for far range (Full body)
        min_detection_confidence: Minimum confidence score [0.0, 1.0]
        min_sharpness_threshold: Minimum variance of Laplacian for blur detection (0 to disable, typical ~50-100)
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_sharpness_threshold = min_sharpness_threshold
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=model_selection, 
            min_detection_confidence=min_detection_confidence
        )

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
        
        # Process the image and find faces
        detection_results = self.face_detection.process(image_rgb)
        
        if detection_results.detections:
            h_img, w_img, _ = frame.shape
            for detection in detection_results.detections:
                score = detection.score[0]
                
                # Manual filtering to ensure strict confidence threshold
                if score < self.min_detection_confidence:
                    continue

                bboxC = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute pixel values
                x = int(bboxC.xmin * w_img)
                y = int(bboxC.ymin * h_img)
                width = int(bboxC.width * w_img)
                height = int(bboxC.height * h_img)

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
