"""
DeepSORT Tracker - ใช้ deep-sort-realtime library
รองรับ Re-ID embedder: mobilenet, clip, torchreid
"""

import numpy as np

class DeepSortTracker:
    def __init__(self, max_age=30, n_init=3, embedder="mobilenet", embedder_gpu=True):
        """
        Initialize DeepSORT Tracker.
        
        Args:
            max_age: จำนวนเฟรมที่รอถ้า Track หายไป (เหมือน track_buffer ใน ByteTrack)
            n_init: จำนวนเฟรมขั้นต่ำที่ต้องเห็นก่อนยืนยัน Track
            embedder: Re-ID model - "mobilenet" (default), "clip", "torchreid"
            embedder_gpu: ใช้ GPU สำหรับ embedder หรือไม่
        """
        print(f"Initializing DeepSORT Tracker (embedder: {embedder})...")
        
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError:
            raise ImportError(
                "deep-sort-realtime not installed. Please run:\n"
                "pip install deep-sort-realtime"
            )
        
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            embedder=embedder,
            embedder_gpu=embedder_gpu
        )
        
        self.embedder = embedder
    
    def update(self, detections_list, frame):
        """
        Update tracker with new detections.
        
        Args:
            detections_list (list): List of dicts from Detectors 
                                    [{'bbox': [x,y,w,h], 'score': 0.9, 'class_id': 0, 'sharpness': 50.0}, ...]
            frame (np.array): Current frame (required for Re-ID embedding)
        
        Returns:
            list: List of tracked objects with 'track_id' added.
        """
        if not detections_list:
            # Still need to update tracker with empty detections
            self.tracker.update_tracks([], frame=frame)
            return []
        
        # Convert to DeepSORT format: [[x, y, w, h, confidence], ...]
        detections_for_deepsort = []
        metadata_list = []
        
        for det in detections_list:
            x, y, w, h = det['bbox']
            score = det['score']
            
            # DeepSORT expects [left, top, width, height, confidence]
            detections_for_deepsort.append(([x, y, w, h], score, 'object'))
            
            # Store metadata including score
            metadata_list.append({
                'class_id': det.get('class_id', 0),
                'sharpness': det.get('sharpness', 0.0),
                'original_bbox': det['bbox'],
                'score': score
            })
        
        # Update DeepSORT
        tracks = self.tracker.update_tracks(detections_for_deepsort, frame=frame)
        
        # Convert back to our format
        results = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            
            x1, y1, x2, y2 = ltrb
            w = x2 - x1
            h = y2 - y1
            
            # Try to match with original detection for metadata
            class_id = 0
            sharpness = 0.0
            # Default score to track's stored confidence or 0.0 if missing
            current_score = track.det_conf if track.det_conf else 0.0
            
            # Find closest detection by IoU
            best_iou = 0
            best_idx = -1
            for i, meta in enumerate(metadata_list):
                ox, oy, ow, oh = meta['original_bbox']
                iou = self._calculate_iou([x1, y1, w, h], [ox, oy, ow, oh])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            # If matched with a detection (IoU > 0.3), use its metadata and score
            if best_idx >= 0 and best_iou > 0.3:
                class_id = metadata_list[best_idx]['class_id']
                sharpness = metadata_list[best_idx]['sharpness']
                current_score = metadata_list[best_idx]['score']
            
            results.append({
                "bbox": [int(x1), int(y1), int(w), int(h)],
                "score": current_score,
                "track_id": int(track_id),
                "class_id": class_id,
                "sharpness": float(sharpness),
                "missed_frames": track.time_since_update
            })
        
        return results
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bboxes in [x, y, w, h] format."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to [x1, y1, x2, y2]
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]
        
        # Intersection
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area
