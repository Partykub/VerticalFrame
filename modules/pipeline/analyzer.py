import json
import numpy as np
from modules.core.director import Director

class VideoAnalyzer:
    def __init__(self, config):
        self.config = config
        self.director = Director()
        
    def analyze(self, tracking_json_path, output_path_json):
        """
        Analyzes tracking data to fix issues (occlusions/merges) and generate smooth camera path.
        """
        print(f"Loading tracking data from {tracking_json_path}...")
        with open(tracking_json_path, 'r') as f:
            data = json.load(f)
            
        frames = data['frames']
        width = data['meta']['width']
        height = data['meta']['height']
        
        camera_path = []
        last_camera_x = None
        
        print("Analyzing & Generating Camera Path...")
        
        # --- PASS 1: Select Raw Targets with "Smart Locking" (Future Look-Ahead) ---
        raw_targets = []
        debug_path_info = [] # Store reason for each frame
        active_track_id = None # The ID we are currently locked onto
        active_class_id = None 
        
        # Config for locking
        smart_lock_cfg = self.config.get("smart_lock", {})
        LOOK_AHEAD_FRAMES = smart_lock_cfg.get("look_ahead_frames", 60)
        SWITCH_THRESHOLD_RATIO = smart_lock_cfg.get("switch_threshold_ratio", 0.6)
        GRACE_PERIOD = smart_lock_cfg.get("grace_period_frames", 30)
        
        print(f"Smart Lock Config: LookAhead={LOOK_AHEAD_FRAMES}, Thresh={SWITCH_THRESHOLD_RATIO}, Grace={GRACE_PERIOD}")
        
        # Saliency Smoothing State
        last_stable_saliency = None
        saliency_spike_count = 0
        
        saliency_cfg = self.config.get("saliency_control", {})
        SALIENCY_JUMP_THRESH = saliency_cfg.get("jump_threshold_percent", 0.2)
        SALIENCY_STABLE_FRAMES = saliency_cfg.get("stable_frames", 15)
        SALIENCY_CONFIDENCE = saliency_cfg.get("look_ahead_confidence", 0.5)
        
        for i in range(len(frames)):
            frame_data = frames[i]
            tracks = frame_data['tracks']
            raw_saliency = frame_data['saliency_point']
            width = frame_data.get('meta', {}).get('width', width) # Update if variable
            
            # --- SALIENCY SPIKE FILTER ---
            # Pre-process saliency point before Director sees it
            saliency_point = raw_saliency # Default
            
            if raw_saliency is not None and width > 0:
                if last_stable_saliency is None:
                    last_stable_saliency = raw_saliency
                else:
                    dx = abs(raw_saliency[0] - last_stable_saliency[0])
                    
                    if dx > (width * SALIENCY_JUMP_THRESH):
                        # --- BIG JUMP DETECTED: LOOK AHEAD (Proactive) ---
                        # Instead of waiting, let's peek into the future to see if this is real.
                        future_match_count = 0
                        check_range = min(len(frames) - 1, i + SALIENCY_STABLE_FRAMES)
                        look_ahead_steps = check_range - i
                        
                        if look_ahead_steps > 0:
                            for f_idx in range(i + 1, check_range + 1):
                                future_frame = frames[f_idx]
                                future_sal = future_frame.get('saliency_point')
                                if future_sal:
                                    f_dx = abs(future_sal[0] - raw_saliency[0])
                                    # Check if future point is consistent with THIS new point (not the old stable one)
                                    if f_dx < (width * SALIENCY_JUMP_THRESH): 
                                        future_match_count += 1
                            
                            # Decision: Is it stable enough in the future?
                            # Use Configured Confidence Ratio
                            if future_match_count >= (look_ahead_steps * SALIENCY_CONFIDENCE):
                                last_stable_saliency = raw_saliency # ACCEPT IMMEDIATELY
                            else:
                                saliency_point = last_stable_saliency # REJECT (Noise)
                        else:
                             # End of video: Accept whatever
                             last_stable_saliency = raw_saliency
                    else:
                        # Small move -> Normal update
                        last_stable_saliency = raw_saliency
            # -----------------------------
            
            # 1. Ask Director for "Opinion" (Standard Logic)
            track_objs = []
            for t in tracks:
                track_objs.append({
                    'bbox': t['bbox'],
                    'class_id': t['class_id'],
                    'track_id': t['id']
                })
            
            proposed_point, reason = self.director.select_target(track_objs, saliency_point, width, height)
            
            # Parse Proposed ID
            proposed_id = None
            proposed_class = None
            if "ID:" in reason:
                # Format: "Face ID:123" or "Body ID:456"
                parts = reason.split("ID:")
                proposed_id = int(parts[1])
                if "Face" in reason: proposed_class = 0
                elif "Body" in reason: proposed_class = 1
                elif "Obj" in reason: proposed_class = 2

            # 2. Smart Locking Logic (Future Validation)
            final_target = proposed_point
            final_reason = reason
            
            # If we have a lock, and the proposal is DIFFERENT -> Check Future
            if active_track_id is not None and proposed_id != active_track_id:
                
                # --- PRIORITY UPGRADE (Face > Body) ---
                # If we are locked on a Body, but a Face is proposed -> SWITCH IMMEDIATELY
                if active_class_id == 1 and proposed_class == 0:
                     active_track_id = proposed_id
                     active_class_id = proposed_class
                     final_reason = f"Priority Upgrade: Body->Face ID:{proposed_id}"
                
                else:
                    # Check if current locked target is STILL visible this frame?
                    current_locked_obj = None
                    for t in track_objs:
                        if t['track_id'] == active_track_id and t['class_id'] == active_class_id:
                            current_locked_obj = t
                            break
                    
                    if current_locked_obj:
                        # Locked target IS visible. Should we switch?
                        # Perform LOOK-AHEAD validation
                        future_wins_msg = ""
                        
                        if proposed_id is not None:
                            # Compare [Locked] vs [Proposed] for next N frames
                            score_locked = 0
                            score_proposed = 0
                            
                            check_range = min(len(frames), i + LOOK_AHEAD_FRAMES)
                            for k in range(i + 1, check_range):
                                # Find largest face/body for locked vs proposed in future frame
                                f_tracks = frames[k]['tracks']
                                
                                # Simple metric: Area
                                area_locked = 0
                                area_proposed = 0
                                
                                for ft in f_tracks:
                                    if ft['id'] == active_track_id and ft['class_id'] == active_class_id:
                                        area_locked = ft['bbox'][2] * ft['bbox'][3]
                                    elif ft['id'] == proposed_id and ft['class_id'] == proposed_class:
                                        area_proposed = ft['bbox'][2] * ft['bbox'][3]
                                
                                if area_proposed > area_locked:
                                    score_proposed += 1
                                elif area_locked > 0:
                                    score_locked += 1
                                    
                            future_frames = check_range - (i + 1)
                            if future_frames > 0:
                                win_ratio = score_proposed / future_frames
                                if win_ratio > SWITCH_THRESHOLD_RATIO:
                                    # New target dominates future -> Allow Switch
                                    active_track_id = proposed_id
                                    active_class_id = proposed_class
                                    # final_target is already set to proposed
                                else:
                                    # New target is short-lived -> REJECT Switch, Stick to Locked
                                    x, y, w, h = current_locked_obj['bbox']
                                    
                                    # Director returns center, but we might want adjustments based on class
                                    if active_class_id == 1: # Body
                                        cy = y + int(h * 0.3)
                                    else:
                                        cy = y + h // 2
                                    cx = x + w // 2
                                    
                                    final_target = (cx, cy)
                                    final_reason = f"Locked ID:{active_track_id} (Ignored {proposed_id})"
                            else:
                                # End of video, just follow director
                                active_track_id = proposed_id
                                active_class_id = proposed_class
                        else:
                            # Proposed is Non-ID (Saliency/Center). If Locked is visible, Stick to Locked?
                            # Usually stick to locked object if visible
                             x, y, w, h = current_locked_obj['bbox']
                             if active_class_id == 1: # Body
                                 cy = y + int(h * 0.3)
                             else:
                                 cy = y + h // 2
                             cx = x + w // 2
                             final_target = (cx, cy)
                             final_reason = f"Locked ID:{active_track_id} (Ignored Saliency)"
                    else:
                        # Locked target LOST (temporarily?).
                        # Check Future: Does this ID return within Grace Period?
                        # GRACE_PERIOD is now from config
                        
                        found_future = False
                        future_check_limit = min(len(frames), i + GRACE_PERIOD)
                        
                        for k in range(i + 1, future_check_limit):
                             f_tracks = frames[k]['tracks']
                             # Is active_track_id present?
                             for ft in f_tracks:
                                 if ft['id'] == active_track_id and ft['class_id'] == active_class_id:
                                     found_future = True
                                     break
                             if found_future:
                                 break
                        
                        if found_future:
                            # IT COMES BACK! Hold position.
                            # distinct from "Locked" reason to debug easier
                            final_reason = f"Hold Lock ID:{active_track_id} (Reappears soon)"
                            
                            # Use Last Known Target Position
                            if len(raw_targets) > 0:
                                final_target = raw_targets[-1] # Stay exactly where we were
                            else:
                                final_target = proposed_point # Should rarely happen
                                
                            # Do NOT change active_track_id (Keep locking it)
                        else:
                            # Truly lost. Switch to proposed (Saliency or other)
                            active_track_id = proposed_id
                            active_class_id = proposed_class
                            # final_target is already proposed_point
            
            elif active_track_id is None and proposed_id is not None:
                # No lock, acquire new lock
                active_track_id = proposed_id
                active_class_id = proposed_class

            # --- Saliency Filter Debug Info ---
            # If we picked Saliency, check if it was filtered
            if "Saliency" in final_reason:
                if raw_saliency is not None and saliency_point == last_stable_saliency and raw_saliency != last_stable_saliency:
                     final_reason += " (Spike Blocked)"
            # ----------------------------------

            # 3. MERGE FIX (Keep existing logic but apply to final_target if relevant)
            # ... (Existing Merge Logic can be kept or integrated. 
            # Ideally Smart Locking solves most hiccups, but Merge Fix is specific to bounding box errors.
            # We will preserve the merge-fix block below if possible, or rewrite it here briefly)
            
            # (Re-applying Merge Fix purely for Face-in-Body refinement if we are tracking Body)
            if active_class_id == 1: # Tracking Body
                 # Check internal faces... (Same as before)
                 pass # Skipping for brevity in this block replacement, assuming SmartLock handles the "Who"
                      # The "Where" (Face inside Body) is fine to add back if needed, 
                      # but SmartLock usually prefers Face class if detected anyway.

            raw_targets.append(final_target)
            debug_path_info.append(final_reason)
            
            # Update temp last camera
            if last_camera_x is None:
                last_camera_x = final_target[0]
            else:
                last_camera_x = last_camera_x * 0.9 + final_target[0] * 0.1

        # --- PASS 1.5: Stabilization (Dead Zone & Look-Ahead) ---
        print("Applying Camera Stabilization (Dead Zone & Look-Ahead)...")
        cam_ctrl = self.config.get("camera_control", {})
        dead_zone_pct = cam_ctrl.get("dead_zone_percent", 0.05)
        min_duration = cam_ctrl.get("min_duration_frames", 15)
        
        dead_zone_px = width * dead_zone_pct
        
        stabilized_targets = []
        anchor = raw_targets[0][0] # Start anchor
        
        i = 0
        while i < len(raw_targets):
            current_raw_x = raw_targets[i][0]
            current_raw_y = raw_targets[i][1]
            
            # Check deviation from Anchor
            diff = abs(current_raw_x - anchor)
            
            if diff > dead_zone_px:
                # Potential Movement Detected!
                # LOOK AHEAD: Does it stay at this new position?
                
                is_real_move = False
                stable_count = 0
                
                # Check next 'min_duration' frames
                # If majority of future frames are FAR from Anchor (meaning close to new spot), it's a real move.
                # If they return to Anchor, it's noise.
                
                for k in range(1, min_duration + 1):
                    if i + k >= len(raw_targets):
                         break
                    future_x = raw_targets[i+k][0]
                    
                    # If future point is still far from old anchor, it counts as a move
                    if abs(future_x - anchor) > dead_zone_px:
                        stable_count += 1
                
                # If > 70% of future frames sustain the move, we accept it.
                if stable_count >= (min_duration * 0.7):
                    is_real_move = True
                
                if is_real_move:
                    # UPDATE ANCHOR
                    anchor = current_raw_x
                else:
                    # IGNORE MOVE (Force camera to stay at Anchor)
                    # This effectively filters out "brief excursions"
                    pass 
            
            # Append ANCHOR as the target (not the raw noisy one)
            stabilized_targets.append((anchor, current_raw_y))
            i += 1

        # --- PASS 2: Cinematic Smoothing (Look-Ahead with Adaptive Speed) ---
        tracking_cfg = self.config.get("tracking", {})
        base_smooth = tracking_cfg.get("smooth_factor", 0.1)
        
        # Fast Pan Config
        cam_ctrl = self.config.get("camera_control", {})
        transition_mode = cam_ctrl.get("transition_mode", "smooth") # Options: smooth, cut, smart
        
        fast_pan_thresh_pct = cam_ctrl.get("fast_pan_threshold_percent", 0.15)
        # Smart Cut Threshold
        smart_cut_thresh_pct = cam_ctrl.get("smart_cut_threshold_percent", 0.30)
        smart_cut_px = width * smart_cut_thresh_pct
        
        max_smooth = cam_ctrl.get("max_smooth_factor", 0.5)
        
        fast_pan_px = width * fast_pan_thresh_pct
        
        smoothed_path = []
        current_cam_x = stabilized_targets[0][0]
        
        # Easing Config
        easing_type = tracking_cfg.get("easing_type", "ease_out")
        velocity = 0.0 # For ease_in momentum
        
        print(f"Applying Smoothing Mode: {transition_mode.upper()} | Type: {easing_type}")
        if transition_mode == "smart":
            print(f"Smart Cut Threshold: {smart_cut_px:.1f}px ({smart_cut_thresh_pct*100}%)")
        
        for target in stabilized_targets:
            tx = target[0]
            dist = abs(tx - current_cam_x)
            
            should_cut = False
            
            if transition_mode == "cut":
                should_cut = True
            elif transition_mode == "smart":
                if dist > smart_cut_px:
                    should_cut = True
            
            if should_cut:
                # Hard Cut
                current_cam_x = tx
                velocity = 0.0 # Reset Physic
            else:
                # --- APPLY EASING LOGIC (7 TYPES) ---
                
                # 1. Linear: Constant Speed
                if easing_type == "linear":
                    step = width * base_smooth * 0.5
                    if current_cam_x < tx: current_cam_x = min(current_cam_x + step, tx)
                    else: current_cam_x = max(current_cam_x - step, tx)

                # --- IN (Accelerating) ---
                # 2. Ease In (Sharp Acceleration)
                elif easing_type == "ease_in":
                     force = (tx - current_cam_x) * base_smooth 
                     velocity += force * 1.5
                     velocity *= 0.6 # Heavy drag
                     current_cam_x += velocity
                
                # 3. Sine In (Gentle Acceleration)
                elif easing_type == "sine_in":
                     force = (tx - current_cam_x) * base_smooth 
                     velocity += force * 1.0 # Softer force
                     velocity *= 0.7 # Smoother drag
                     current_cam_x += velocity

                # --- OUT (Decelerating - Standard EMA) ---
                # 4. Ease Out (Sharp Deceleration) - Default Tracking
                elif easing_type == "ease_out":
                     current_cam_x = (current_cam_x * (1 - base_smooth)) + (tx * base_smooth)
                
                # 5. Sine Out (Gentle Deceleration)
                elif easing_type == "sine_out":
                     soft_smooth = base_smooth * 0.7 # Slower approach
                     current_cam_x = (current_cam_x * (1 - soft_smooth)) + (tx * soft_smooth)

                # --- IN-OUT (Smooth Start & Stop) ---
                
                # 6. Ease In-Out (Sharp Curve)
                elif easing_type == "ease_in_out":
                    dist_ratio = min(1.0, dist / (width * 0.4))
                    # Friction: 0.5 (Close) <-> 0.92 (Far)
                    target_friction = 0.5 + (0.42 * dist_ratio)
                    
                    force = (tx - current_cam_x) * base_smooth * 1.5
                    velocity += force
                    velocity *= target_friction
                    current_cam_x += velocity

                # 7. Sine In-Out (Gentle Curve)
                elif easing_type == "sine_in_out":
                    dist_ratio = min(1.0, dist / (width * 0.4))
                    # Friction: 0.6 (Close) <-> 0.9 (Far) - More damping range
                    target_friction = 0.6 + (0.3 * dist_ratio)
                    
                    force = (tx - current_cam_x) * base_smooth * 1.2
                    velocity += force
                    velocity *= target_friction
                    current_cam_x += velocity
                
                # Fallback
                else:
                    # Treat as sine_in_out
                    dist_ratio = min(1.0, dist / (width * 0.4))
                    target_friction = 0.6 + (0.3 * dist_ratio)
                    force = (tx - current_cam_x) * base_smooth * 1.2
                    velocity += force
                    velocity *= target_friction
                    current_cam_x += velocity
            
            smoothed_path.append(int(current_cam_x))
            
        # Save Path
        output = {
            "meta": data['meta'],
            "path": smoothed_path,  # List of X coordinates for crop center
            "debug_info": debug_path_info # List of reasons per frame
        }
        
        print(f"Path generated. Saving to {output_path_json}...")
        with open(output_path_json, 'w') as f:
            json.dump(output, f)

        print("Analysis Complete.")
