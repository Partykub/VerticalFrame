import numpy as np

class Director:
    def __init__(self):
        self.current_target = None # (x, y)
        print("Director initialized: Priorities = [Face > Body > Saliency]")

    def select_target(self, tracked_objects, saliency_point, frame_width, frame_height):
        """
        Choose the best target point based on priority logic.
        
        Args:
            tracked_objects (list): List of dicts {'bbox':.., 'class_id':..}
            saliency_point (tuple): (x, y) from saliency detector
            frame_width, frame_height: Dimensions of the frame
            
        Returns:
            tuple: (target_x, target_y) - The chosen center point for the camera.
            str: reason - Description of what was chosen (e.g., "Face ID:1")
        """
        
        # Priority 1: Face (Class ID 0)
        faces = [obj for obj in tracked_objects if obj.get('class_id') == 0]
        if faces:
            # Logic: Choose the largest face (assuming it's the main subject)
            # We could also add logic for "central face" or "speaking face" later
            best_face = max(faces, key=lambda o: o['bbox'][2] * o['bbox'][3]) # Area = w*h
            
            x, y, w, h = best_face['bbox']
            center_x = x + w // 2
            center_y = y + h // 2
            return (center_x, center_y), f"Face ID:{best_face['track_id']}"

        # Priority 2: Body (Class ID 1)
        bodies = [obj for obj in tracked_objects if obj.get('class_id') == 1]
        if bodies:
            best_body = max(bodies, key=lambda o: o['bbox'][2] * o['bbox'][3])
            
            x, y, w, h = best_body['bbox']
            center_x = x + w // 2
            # For body, we usually want to frame slightly higher than center (Head & Shoulders bias)
            # Let's say at 30% from top of bbox instead of 50%
            center_y = y + int(h * 0.3) 
            
            return (center_x, center_y), f"Body ID:{best_body['track_id']}"
            
        # Priority 3: Other Objects (Class ID 2)
        objects = [obj for obj in tracked_objects if obj.get('class_id') == 2]
        if objects:
            best_obj = max(objects, key=lambda o: o['bbox'][2] * o['bbox'][3])
            x, y, w, h = best_obj['bbox']
            center_x = x + w // 2
            center_y = y + h // 2
            return (center_x, center_y), f"Obj ID:{best_obj['track_id']}"

        # Priority 4: Saliency
        if saliency_point:
            return saliency_point, "Saliency"

        # Fallback: Center of frame
        return (frame_width // 2, frame_height // 2), "Center (Default)"
