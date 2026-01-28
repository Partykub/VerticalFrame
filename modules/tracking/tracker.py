import numpy as np
import supervision as sv

class ByteTracker:
    def __init__(self, track_thresh=0.25, track_buffer=30, match_thresh=0.8):
        """
        Initialize ByteTrack using Supervision library.
        
        Args:
            track_thresh (float): Detection confidence threshold to activate tracking (default 0.25)
            track_buffer (int): Number of frames to keep a lost track alive (default 30)
            match_thresh (float): Threshold for matching tracks (default 0.8)
        """
        print("Initializing ByteTracker (via supervision)...")
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh
        )

    def update(self, detections_list):
        """
        Update tracker with new detections.
        
        Args:
            detections_list (list): List of dicts from Detectors 
                                    [{'bbox': [x,y,w,h], 'score': 0.9, 'class_id': 0}, ...]
        
        Returns:
            list: List of tracked objects with 'track_id' added.
        """
        if not detections_list:
            empty_detections = sv.Detections.empty()
            tracked_detections = self.tracker.update_with_detections(empty_detections)
            return []

        # 1. Convert format (list of dicts) to supervision.Detections
        xyxy = []
        confidence = []
        class_id = []

        for det in detections_list:
            x, y, w, h = det['bbox']
            score = det['score']
            cls = det.get('class_id', 0) # Default to 0 if not present
            
            # Convert xywh to xyxy
            xyxy.append([x, y, x + w, y + h])
            confidence.append(score)
            class_id.append(cls)

        detections = sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidence),
            class_id=np.array(class_id)
        )

        # 2. Update ByteTrack
        tracked_detections = self.tracker.update_with_detections(detections)
        
        # 3. Convert back to list of dicts
        results = []
        
        if tracked_detections.tracker_id is None:
             return []

        for i, tracker_id in enumerate(tracked_detections.tracker_id):
            x1, y1, x2, y2 = tracked_detections.xyxy[i]
            w = x2 - x1
            h = y2 - y1
            conf = tracked_detections.confidence[i] if tracked_detections.confidence is not None else 0.0
            cls_id = int(tracked_detections.class_id[i]) if tracked_detections.class_id is not None else 0
            
            # Determine type label
            obj_type = "face" # Default
            if cls_id == 0:
                 # Note: YOLO person is 0, but MediaPipe face we might also map to 0? 
                 # We must coordinate Class IDs in the pipeline.
                 # Let's assume Pipeline sends: Face=0, Person=1, Others=2+
                 pass 
            
            results.append({
                "bbox": [int(x1), int(y1), int(w), int(h)],
                "score": float(conf),
                "track_id": int(tracker_id),
                "class_id": cls_id
                # 'type' will be determined by pipeline based on class_id or passed through
            })
            
        return results
