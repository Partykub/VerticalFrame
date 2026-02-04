import numpy as np

class Director:
    def __init__(self):
        self.current_target = None # (x, y)
        self.last_selected_id = None  # Track last selected body/face for sticky selection
        self.last_selected_class = None
        print("Director initialized: Priorities = [Face > Body > Saliency]")

    def select_target(self, tracked_objects, saliency_point, frame_width, frame_height, future_frames=None, config=None):
        """
        Choose the best target point based on priority logic.
        
        Args:
            tracked_objects (list): List of dicts {'bbox':.., 'class_id':..}
            saliency_point (tuple): (x, y) from saliency detector
            frame_width, frame_height: Dimensions of the frame
            future_frames (list): Optional list of future frame data for look-ahead
            config (dict): Optional director config
            
        Returns:
            tuple: (target_x, target_y) - The chosen center point for the camera.
            str: reason - Description of what was chosen (e.g., "Face ID:1")
        """
        
        # Priority 1: Face (Class ID 0)
        faces = [obj for obj in tracked_objects if obj.get('class_id') == 0]
        if faces:
            # Filter out low-confidence faces (likely false positives like walls, patterns)
            MIN_FACE_CONFIDENCE = 0.5  # MediaPipe confidence threshold
            high_confidence_faces = [f for f in faces if f.get('conf', 0) >= MIN_FACE_CONFIDENCE]
            
            if high_confidence_faces:
                # Sticky selection: If last selected face still exists, prefer it unless new one is significantly larger
                best_face = None
                
                # Helper function for scoring
                def get_score(face):
                    x, y, w, h = face['bbox']
                    area = w * h
                    
                    # 1. Size Score (Power 0.5 = Sqrt) -> ลดอิทธิพลของหน้าใหญ่ลงอีก (จากเดิม 0.6)
                    size_score = area ** 0.5
                    
                    # 2. Centrality Penalty (Very Aggressive for OTS shots)
                    center_x = x + (w / 2)
                    frame_center = frame_width / 2
                    dist_norm = abs(center_x - frame_center) / (frame_width / 2) 
                    
                    # Penalty: 1.5 Multiplier (หน้าขอบจอ คะแนนแทบเป็น 0)
                    position_factor = 1.0 - (dist_norm * 1.5)
                    if position_factor < 0.05: position_factor = 0.05
                    
                    return size_score * position_factor

                if self.last_selected_id is not None and self.last_selected_class == 0:
                    # Check if last selected face is still present
                    last_face = next((f for f in high_confidence_faces if f['track_id'] == self.last_selected_id), None)
                    
                    if last_face:
                        best_candidate = max(high_confidence_faces, key=get_score)
                        
                        last_score = get_score(last_face)
                        best_score = get_score(best_candidate)
                        
                        # DEBUG PRINT
                        print(f"DEBUG: Last(ID:{last_face['track_id']}) Score={last_score:.2f} | New(ID:{best_candidate['track_id']}) Score={best_score:.2f}")

                        # SWITCH LOGIC:
                        # Read threshold from config (1.0=switch immediately, 1.1=10% better required)
                        base_threshold = config.get('face_switch_threshold', 1.1) if config else 1.1
                        threshold = base_threshold
                        
                        # Rescue: ถ้าคนเดิมคะแนนแย่มาก (ตกขอบ) ให้เปลี่ยนง่ายขึ้น
                        if last_score < (best_score * 0.5): 
                             threshold = 1.0
                        
                        if best_score > last_score * threshold:
                            print(f"DEBUG: >>> SWITCHING to ID:{best_candidate['track_id']}")
                            best_face = best_candidate
                        else:
                            best_face = last_face  # Stick to current
                    else:
                        best_face = max(high_confidence_faces, key=get_score)
                else:
                    best_face = max(high_confidence_faces, key=get_score)
                
                x, y, w, h = best_face['bbox']
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Remember this selection
                self.last_selected_id = best_face['track_id']
                self.last_selected_class = 0
                
                return (center_x, center_y), f"Face ID:{best_face['track_id']}"
            else:
                # All faces are low-confidence (false positives)
                # Fall through to Body detection
                pass

        # Priority 2: Body (Class ID 1)
        bodies = [obj for obj in tracked_objects if obj.get('class_id') == 1]
        if bodies:
            # ===== LOOK-AHEAD: Check if Face will appear soon =====
            # ถ้ามี Face ในอนาคตใกล้ๆ Body → ข้าม Body แล้วใช้ตำแหน่ง Face แทน
            if future_frames and config:
                look_ahead = config.get('body_to_face_lookahead', 15)
                overlap_thresh = config.get('body_face_overlap_threshold', 0.3)
                
                # หา Body ที่ใหญ่ที่สุด (ที่กำลังจะ Focus)
                largest_body = max(bodies, key=lambda o: o['bbox'][2] * o['bbox'][3])
                bx, by, bw, bh = largest_body['bbox']
                body_center_x = bx + bw // 2
                
                # ดูอนาคต N เฟรม
                for future_frame in future_frames[:look_ahead]:
                    future_tracks = future_frame.get('tracks', [])
                    future_faces = [t for t in future_tracks if t.get('class_id') == 0]
                    
                    for face in future_faces:
                        fx, fy, fw, fh = face['bbox']
                        face_center_x = fx + fw // 2
                        
                        # เช็คว่า Face อยู่ใกล้ Body ไหม (X-axis overlap)
                        x_overlap = abs(face_center_x - body_center_x) < (bw * overlap_thresh + fw * overlap_thresh)
                        
                        if x_overlap:
                            # พบ Face ในอนาคต → ใช้ตำแหน่ง Face แทน Body
                            center_x = face_center_x
                            center_y = fy + fh // 2
                            
                            # Remember this selection (as Face)
                            self.last_selected_id = face.get('id', face.get('track_id'))
                            self.last_selected_class = 0
                            
                            print(f"  [Director] Skip Body, use future Face #{self.last_selected_id}")
                            return (center_x, center_y), f"Face ID:{self.last_selected_id} (Future)"
            
            # ===== ไม่พบ Face ในอนาคต → ใช้ Body ตามปกติ =====
            # Sticky selection for bodies too
            best_body = None
            
            if self.last_selected_id is not None and self.last_selected_class == 1:
                # Check if last selected body is still present
                last_body = next((b for b in bodies if b['track_id'] == self.last_selected_id), None)
                
                if last_body:
                    # Compare with largest body
                    largest_body = max(bodies, key=lambda o: o['bbox'][2] * o['bbox'][3])
                    
                    last_area = last_body['bbox'][2] * last_body['bbox'][3]
                    largest_area = largest_body['bbox'][2] * largest_body['bbox'][3]
                    
                    # Only switch if new body is 1.5x larger
                    if largest_area > last_area * 1.5:
                        best_body = largest_body
                    else:
                        best_body = last_body  # Stick to current
                else:
                    # Last body gone, choose largest
                    best_body = max(bodies, key=lambda o: o['bbox'][2] * o['bbox'][3])
            else:
                # No previous selection, choose largest
                best_body = max(bodies, key=lambda o: o['bbox'][2] * o['bbox'][3])
            
            x, y, w, h = best_body['bbox']
            center_x = x + w // 2
            # For body, we usually want to frame slightly higher than center (Head & Shoulders bias)
            # Let's say at 30% from top of bbox instead of 50%
            center_y = y + int(h * 0.3) 
            
            # Remember this selection
            self.last_selected_id = best_body['track_id']
            self.last_selected_class = 1
            
            return (center_x, center_y), f"Body ID:{best_body['track_id']}"
            
        # Priority 3: Other Objects (Class ID 2)
        objects = [obj for obj in tracked_objects if obj.get('class_id') == 2]
        if objects:
            best_obj = max(objects, key=lambda o: o['bbox'][2] * o['bbox'][3])
            x, y, w, h = best_obj['bbox']
            center_x = x + w // 2
            center_y = y + h // 2
            return (center_x, center_y), f"Obj ID:{best_obj['track_id']}"

        # Priority 4: Saliency (only if no other option and we haven't locked to anything recently)
        # Avoid frequent switching to/from saliency
        if saliency_point:
            # Only use saliency if we've been without a target for a while
            # (to prevent flickering between Body and Saliency)
            if self.last_selected_id is None:
                return saliency_point, "Saliency"
            else:
                # Had a previous target, prefer to hold center rather than jump to saliency
                return (frame_width // 2, frame_height // 2), "Center (No Target)"

        # Fallback: Center of frame
        return (frame_width // 2, frame_height // 2), "Center (Default)"
