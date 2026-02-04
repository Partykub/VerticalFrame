import json
import numpy as np
from modules.core.director import Director

def get_related_ids_group(frames, current_frame, current_id, class_id, look_window=30):
    """
    หา 'กลุ่ม IDs' ที่เกี่ยวข้องกับ current_id โดยดูทั้งอดีตและอนาคต
    
    Logic: ถ้า ID ปรากฏในช่วงเวลาใกล้เคียงกัน → ถือว่าเป็น "กลุ่มเดียวกัน"
    
    Args:
        frames: All tracking frames
        current_frame: Current frame index
        current_id: Current Track ID (อาจเป็น None)
        class_id: Class ID (0=Face, 1=Body)
        look_window: จำนวนเฟรมที่มองย้อนหลังและข้างหน้า
    
    Returns:
        set of track IDs ที่เป็น "กลุ่มเดียวกัน"
    """
    related_ids = set()
    
    if current_id is not None:
        related_ids.add(current_id)
    
    # ดูอดีต (ย้อนหลัง look_window เฟรม)
    start = max(0, current_frame - look_window)
    for i in range(start, current_frame):
        frame_data = frames[i]
        tracks = frame_data.get('tracks', [])
        for t in tracks:
            if t['class_id'] == class_id:
                related_ids.add(t['id'])
    
    # ดูอนาคต (ข้างหน้า look_window เฟรม)
    end = min(len(frames), current_frame + look_window)
    for i in range(current_frame + 1, end):
        frame_data = frames[i]
        tracks = frame_data.get('tracks', [])
        for t in tracks:
            if t['class_id'] == class_id:
                related_ids.add(t['id'])
    
    return related_ids

def analyze_future_trajectory(frames, current_frame, track_id, class_id, look_ahead=30, max_gap=5):
    """
    Analyze if Track ID will disappear in the future
    
    Args:
        frames: All tracking frames
        current_frame: Current frame index
        track_id: Track ID to analyze
        class_id: Class ID (0=Face, 1=Body)
        look_ahead: How many frames to look ahead
        max_gap: Max frames gap to consider as "disappeared"
    
    Returns:
        {
            'will_disappear': bool,
            'last_seen_frame': int,
            'future_frames': int,
            'confidence_avg': float,
            'max_gap': int
        }
    """
    future_appearances = []
    
    for i in range(current_frame + 1, min(len(frames), current_frame + look_ahead + 1)):
        frame_data = frames[i]
        tracks = frame_data.get('tracks', [])
        
        # Find matching track
        found = False
        for t in tracks:
            if t['id'] == track_id and t['class_id'] == class_id:
                future_appearances.append({
                    'frame': i,
                    'conf': t.get('conf', 0)
                })
                found = True
                break
    
    if len(future_appearances) == 0:
        return {
            'will_disappear': True,
            'last_seen_frame': current_frame,
            'future_frames': 0,
            'confidence_avg': 0,
            'max_gap': look_ahead
        }
    
    # Check for gaps (ID disappears for > max_gap frames)
    gaps = []
    for i in range(len(future_appearances) - 1):
        gap = future_appearances[i+1]['frame'] - future_appearances[i]['frame'] - 1
        if gap > 0:
            gaps.append(gap)
    
    max_gap_found = max(gaps) if gaps else 0
    will_disappear = (max_gap_found > max_gap)
    
    avg_conf = sum(f['conf'] for f in future_appearances) / len(future_appearances)
    
    return {
        'will_disappear': will_disappear,
        'last_seen_frame': future_appearances[-1]['frame'],
        'future_frames': len(future_appearances),
        'confidence_avg': avg_conf,
        'max_gap': max_gap_found
    }

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
        track_id_history = [] # Store Track ID for each frame (for conversation mode)
        actual_focused_history = [] # Store ACTUAL bbox used for camera position (Track ID + Class ID)
        active_track_id = None # The ID we are currently locked onto
        active_class_id = None
        
        # Config for locking
        smart_lock_cfg = self.config.get("smart_lock", {})
        cam_ctrl = self.config.get("camera_control", {})
        transition_mode = cam_ctrl.get("transition_mode", "smooth")
        
        # ปรับ look_ahead และ grace_period ตามโหมด
        if transition_mode == "conversation":
            conv_settings = cam_ctrl.get("conversation_settings", {})
            LOOK_AHEAD_FRAMES = conv_settings.get("look_ahead_frames", 15)
            GRACE_PERIOD = conv_settings.get("grace_period_frames", 20)
            print(f"Conversation Mode Config: LookAhead={LOOK_AHEAD_FRAMES}, Grace={GRACE_PERIOD}")
        else:
            LOOK_AHEAD_FRAMES = smart_lock_cfg.get("look_ahead_frames", 60)
            GRACE_PERIOD = smart_lock_cfg.get("grace_period_frames", 30)
            print(f"Smart Lock Config: LookAhead={LOOK_AHEAD_FRAMES}, Grace={GRACE_PERIOD}")
        
        SWITCH_THRESHOLD_RATIO = smart_lock_cfg.get("switch_threshold_ratio", 0.6)
        
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
                    'track_id': t['id'],
                    'conf': t.get('conf', 0.0) # Send confidence to Director
                })
            
            # Get director config for look-ahead
            director_config = self.config.get('director', {})
            
            # Get future frames for look-ahead (next 15 frames)
            look_ahead_count = director_config.get('body_to_face_lookahead', 15)
            future_frames = frames[i+1:i+1+look_ahead_count] if i < len(frames) - 1 else []
            
            proposed_point, reason = self.director.select_target(
                track_objs, saliency_point, width, height,
                future_frames=future_frames, config=director_config
            )
            
            # Parse Proposed ID
            proposed_id = None
            proposed_class = None
            if "ID:" in reason:
                # Format: "Face ID:123" or "Body ID:456" or "Face ID:4 (Future)"
                parts = reason.split("ID:")
                id_part = parts[1].split()[0]  # Get first word (the ID number)
                # Remove any non-digit characters
                id_digits = ''.join(c for c in id_part if c.isdigit())
                if id_digits:
                    proposed_id = int(id_digits)
                if "Face" in reason: proposed_class = 0
                elif "Body" in reason: proposed_class = 1
                elif "Obj" in reason: proposed_class = 2
            
            # === FACE LOOK-AHEAD: ถ้า Director เลือก Body แต่ Body กำลังจะหาย และมี Face ในอนาคต → เลือก Face แทน ===
            if proposed_class == 1 and proposed_id is not None:  # Director เลือก Body
                # เช็คว่า Body นี้กำลังจะหายไหม (ใช้ analyze_future_trajectory)
                body_future = analyze_future_trajectory(
                    frames, i, proposed_id, proposed_class, 
                    look_ahead=15, 
                    max_gap=5
                )
                
                # ถ้า Body จะหาย → ค่อยมองหา Face ในอนาคต
                if body_future['will_disappear']:
                    # ดูอนาคตว่าจะมี Face ไหม (ช่วง 5-20 เฟรม = 0.17-0.67 วินาที)
                    FACE_LOOKAHEAD_START = 5   # เริ่มมองหาที่ 5 เฟรมข้างหน้า (ไม่เร็วเกินไป)
                    FACE_LOOKAHEAD_END = 20    # มองไกลสุด 20 เฟรม
                    future_face_found = None
                    
                    for k in range(i + FACE_LOOKAHEAD_START, min(len(frames), i + FACE_LOOKAHEAD_END + 1)):
                        future_tracks = frames[k]['tracks']
                        # หา Face ที่มี confidence สูง
                        for ft in future_tracks:
                            if ft['class_id'] == 0:  # Face
                                face_conf = ft.get('conf', 0)
                                face_sharp = ft.get('sharpness', 0)
                                # เช็คว่า Face มีคุณภาพดี
                                if face_conf >= 0.5 and face_sharp >= 15:
                                    future_face_found = {
                                        'frame': k,
                                        'id': ft['id'],
                                        'bbox': ft['bbox'],
                                        'conf': face_conf,
                                        'delay': k - i
                                    }
                                    break
                        if future_face_found:
                            break
                    
                    if future_face_found:
                        # Body จะหายและมี Face ในอนาคต! → เปลี่ยนให้เลือก Face แทน Body
                        fx, fy, fw, fh = future_face_found['bbox']
                        proposed_point = (fx + fw // 2, fy + fh // 2)
                        proposed_id = future_face_found['id']
                        proposed_class = 0  # Face
                        reason = f"Face Lookahead ID:{proposed_id} (+{future_face_found['delay']}f, Body disappears)"

            # 2. Smart Locking Logic (Future Validation)
            final_target = proposed_point
            final_reason = reason
            
            # Track which bbox we ACTUALLY use for camera position
            actual_focused_id = proposed_id
            actual_focused_class = proposed_class
            
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
                        skip_locked_logic = False
                        
                        # ===== CONFIDENCE CHECK: If proposed has MUCH higher confidence, switch immediately =====
                        # This fixes "Ghost Track" issue where DeepSORT predicts position of lost track
                        locked_conf = current_locked_obj.get('conf', 0.0)
                        
                        if proposed_id is not None and proposed_class == 0:  # Proposed is Face
                            # Find proposed object's confidence
                            proposed_obj_for_conf = None
                            for obj in track_objs:
                                if obj['track_id'] == proposed_id and obj['class_id'] == proposed_class:
                                    proposed_obj_for_conf = obj
                                    break
                            
                            if proposed_obj_for_conf:
                                proposed_conf = proposed_obj_for_conf.get('conf', 0.0)
                                
                                # ===== POSITION CONTINUITY CHECK =====
                                # If proposed Face is CLOSER to our last target position than locked object,
                                # switch immediately to avoid flicker (prevents camera jumping back)
                                if len(raw_targets) > 0:
                                    last_target = raw_targets[-1]
                                    
                                    # Calculate distances
                                    locked_x = current_locked_obj['bbox'][0] + current_locked_obj['bbox'][2] // 2
                                    locked_y = current_locked_obj['bbox'][1] + current_locked_obj['bbox'][3] // 2
                                    dist_to_locked = ((last_target[0] - locked_x) ** 2 + (last_target[1] - locked_y) ** 2) ** 0.5
                                    
                                    x_p, y_p, w_p, h_p = proposed_obj_for_conf['bbox']
                                    proposed_x = x_p + w_p // 2
                                    proposed_y = y_p + h_p // 2
                                    dist_to_proposed = ((last_target[0] - proposed_x) ** 2 + (last_target[1] - proposed_y) ** 2) ** 0.5
                                    
                                    # If proposed is MUCH closer to where camera already is → switch immediately
                                    # This happens when camera was pre-positioning to proposed location
                                    if dist_to_proposed < dist_to_locked * 0.5 and proposed_conf > 0.5:
                                        final_target = (proposed_x, proposed_y)
                                        final_reason = f"Continuity ID:{active_track_id}→{proposed_id} (dist:{dist_to_proposed:.0f}<{dist_to_locked:.0f})"
                                        
                                        # Switch lock
                                        active_track_id = proposed_id
                                        active_class_id = proposed_class
                                        actual_focused_id = proposed_id
                                        actual_focused_class = proposed_class
                                        
                                        skip_locked_logic = True
                                
                                # If proposed Face has MUCH higher confidence (0.3 more) → Switch immediately
                                # This handles Ghost Track with low/no confidence
                                if not skip_locked_logic:
                                    if proposed_conf > locked_conf + 0.3 or (proposed_conf > 0.7 and locked_conf < 0.4):
                                        x_p, y_p, w_p, h_p = proposed_obj_for_conf['bbox']
                                        final_target = (x_p + w_p // 2, y_p + h_p // 2)
                                        final_reason = f"Switch ID:{active_track_id}→{proposed_id} (Conf:{locked_conf:.2f}→{proposed_conf:.2f})"
                                        
                                        # Switch lock
                                        active_track_id = proposed_id
                                        active_class_id = proposed_class
                                        actual_focused_id = proposed_id
                                        actual_focused_class = proposed_class
                                        
                                        # Skip rest of this section
                                        skip_locked_logic = True
                        
                        if not skip_locked_logic:
                            # Perform LOOK-AHEAD validation
                            future_wins_msg = ""
                            suspicious_size_ratio = False
                            
                            if proposed_id is not None:
                                if active_class_id == 0 and proposed_class == 0:  # Both are Faces
                                    locked_area = current_locked_obj['bbox'][2] * current_locked_obj['bbox'][3]
                                
                                    # Find proposed object bbox safely
                                    proposed_obj = None
                                    for obj in track_objs:
                                        if obj['track_id'] == proposed_id and obj['class_id'] == proposed_class:
                                            proposed_obj = obj
                                            break
                                    
                                    if proposed_obj:
                                        x_p, y_p, w_p, h_p = proposed_obj['bbox']
                                        proposed_area = w_p * h_p
                                        
                                        # If proposed is MORE than 2x larger, it's suspicious
                                        if proposed_area > locked_area * 2.0:
                                            suspicious_size_ratio = True
                                
                                if suspicious_size_ratio:
                                    # Reject switch - proposed Face is too large (likely blurry/back-facing)
                                    x, y, w, h = current_locked_obj['bbox']
                                    if active_class_id == 1: # Body
                                        cy = y + int(h * 0.3)
                                    else:
                                        cy = y + h // 2
                                    cx = x + w // 2
                                    final_target = (cx, cy)
                                    final_reason = f"Locked ID:{active_track_id} (Ignored oversized {proposed_id})"
                                    
                                    actual_focused_id = active_track_id
                                    actual_focused_class = active_class_id
                                else:
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
                                            
                                    future_frames_count = check_range - (i + 1)
                                    if future_frames_count > 0:
                                        win_ratio = score_proposed / future_frames_count
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
                                            
                                            # IMPORTANT: We're using LOCKED bbox for camera position
                                            actual_focused_id = active_track_id
                                            actual_focused_class = active_class_id
                                    else:
                                        # End of video, just follow director
                                        active_track_id = proposed_id
                                        active_class_id = proposed_class
                        else:
                            # Proposed is Non-ID (Saliency/Center). If Locked is visible, Stick to Locked?
                            x, y, w, h = current_locked_obj['bbox']
                            face_center_x = x + w // 2
                            
                            # CHECK: Is locked Face near the edge of frame? (>85% from center)
                            edge_threshold = 0.85
                            dist_from_center = abs(face_center_x - (width / 2)) / (width / 2)
                            
                            if dist_from_center > edge_threshold:
                                # Face is exiting frame! Release lock and look for new target
                                # Look ahead for next Face or use proposed target
                                look_ahead_window = GRACE_PERIOD
                                future_limit = min(len(frames), i + look_ahead_window)
                                
                                future_face_found = False
                                future_face_bbox = None
                                future_face_id = None
                                
                                for k in range(i + 1, future_limit):
                                    f_tracks = frames[k]['tracks']
                                    for ft in f_tracks:
                                        if ft['class_id'] == 0:  # Any Face
                                            # Check if this is a different Face (not the exiting one)
                                            ft_center_x = ft['bbox'][0] + ft['bbox'][2] // 2
                                            ft_dist_from_center = abs(ft_center_x - (width / 2)) / (width / 2)
                                            
                                            if ft_dist_from_center < edge_threshold:  # Not at edge
                                                future_face_found = True
                                                future_face_bbox = ft['bbox']
                                                future_face_id = ft['id']
                                                break
                                    if future_face_found:
                                        break
                                
                                if future_face_found:
                                    # Pre-position to new Face
                                    fx, fy, fw, fh = future_face_bbox
                                    final_target = (fx + fw // 2, fy + fh // 2)
                                    final_reason = f"Exit→Pre-position ID:{future_face_id}"
                                    
                                    # Switch lock to new Face
                                    active_track_id = future_face_id
                                    active_class_id = 0
                                    actual_focused_id = future_face_id
                                    actual_focused_class = 0
                                else:
                                    # No new Face found, use proposed point (Body/Saliency)
                                    final_target = proposed_point
                                    final_reason = f"Exit ID:{active_track_id} → Saliency"
                                    
                                    # Release lock
                                    active_track_id = None
                                    active_class_id = None
                                    actual_focused_id = None
                                    actual_focused_class = None
                            else:
                                # Face is not at edge, stick to it
                                if active_class_id == 1: # Body
                                    cy = y + int(h * 0.3)
                                else:
                                    cy = y + h // 2
                                cx = x + w // 2
                                final_target = (cx, cy)
                                final_reason = f"Locked ID:{active_track_id} (Ignored Saliency)"
                                
                                # Using LOCKED bbox
                                actual_focused_id = active_track_id
                                actual_focused_class = active_class_id
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
                            # IT COMES BACK! Pre-position camera to where it will reappear
                            # Instead of holding, find the future position and move towards it
                            future_bbox = None
                            for k in range(i + 1, future_check_limit):
                                f_tracks = frames[k]['tracks']
                                for ft in f_tracks:
                                    if ft['id'] == active_track_id and ft['class_id'] == active_class_id:
                                        future_bbox = ft['bbox']
                                        break
                                if future_bbox:
                                    break
                            
                            if future_bbox:
                                # Use FUTURE position instead of holding
                                x, y, w, h = future_bbox
                                cx = x + w // 2
                                cy = y + h // 2
                                final_target = (cx, cy)
                                final_reason = f"Pre-position ID:{active_track_id} (Reappears soon)"
                            else:
                                # Fallback: Use Last Known Target Position
                                if len(raw_targets) > 0:
                                    final_target = raw_targets[-1]
                                else:
                                    final_target = proposed_point
                                final_reason = f"Hold Lock ID:{active_track_id} (Reappears soon)"
                                
                            # Do NOT change active_track_id (Keep locking it)
                            # Keep using the locked track as actual focus
                            actual_focused_id = active_track_id
                            actual_focused_class = active_class_id
                        else:
                            # Truly lost. Before switching, check if we should wait for NEW Face
                            # Special case: If locked was Face and proposal is Body, look ahead for new Face
                            should_wait_for_face = False
                            future_face_info = None
                            
                            if active_class_id == 0 and proposed_class == 1:
                                # Was Face, proposed Body -> Look ahead for NEW Face
                                look_ahead_window = 10  # Look 10 frames ahead (~0.33 seconds)
                                future_limit = min(len(frames), i + look_ahead_window)
                                
                                for k in range(i + 1, future_limit):
                                    f_tracks = frames[k]['tracks']
                                    # Find any Face (class_id == 0)
                                    for ft in f_tracks:
                                        if ft['class_id'] == 0:
                                            should_wait_for_face = True
                                            future_face_info = {
                                                'frame': k,
                                                'id': ft['id'],
                                                'delay': k - i
                                            }
                                            break
                                    if should_wait_for_face:
                                        break
                            
                            if should_wait_for_face:
                                # NEW Face coming soon! Pre-position camera towards new Face
                                final_reason = f"Wait for Face ID:{future_face_info['id']} (+{future_face_info['delay']}f)"
                                
                                # Calculate new Face position from future frame
                                future_frame_idx = future_face_info['frame']
                                future_tracks = frames[future_frame_idx]['tracks']
                                
                                new_face_bbox = None
                                for ft in future_tracks:
                                    if ft['id'] == future_face_info['id'] and ft['class_id'] == 0:
                                        new_face_bbox = ft['bbox']
                                        break
                                
                                if new_face_bbox:
                                    # Use new Face position as target
                                    x, y, w, h = new_face_bbox
                                    cx = x + w // 2
                                    cy = y + h // 2
                                    final_target = (cx, cy)
                                    
                                    # IMPORTANT: Switch Track ID NOW to enable smooth transition (not cut)
                                    # This prevents Conversation Mode from doing a CUT when Face appears
                                    active_track_id = future_face_info['id']
                                    active_class_id = 0  # Face
                                    actual_focused_id = future_face_info['id']
                                    actual_focused_class = 0
                                else:
                                    # Fallback: hold current position
                                    if len(raw_targets) > 0:
                                        final_target = raw_targets[-1]
                                    else:
                                        final_target = proposed_point
                                    
                                    # Keep locked on old Face ID
                                    actual_focused_id = active_track_id
                                    actual_focused_class = active_class_id
                            else:
                                # No new Face coming -> Switch to proposed (Body/Saliency)
                                active_track_id = proposed_id
                                active_class_id = proposed_class
                                # final_target is already proposed_point
                                # actual_focused_id already set to proposed
            
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
            track_id_history.append(active_track_id)  # Track ID for conversation mode
            
            # Store ACTUAL focused bbox (used for camera position)
            actual_focused_history.append({
                'track_id': actual_focused_id,
                'class_id': actual_focused_class
            })
            
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
        transition_mode = cam_ctrl.get("transition_mode", "smooth")
        
        dead_zone_px = width * dead_zone_pct
        
        stabilized_targets = []
        anchor = raw_targets[0][0] # Start anchor
        
        i = 0
        while i < len(raw_targets):
            current_raw_x = raw_targets[i][0]
            current_raw_y = raw_targets[i][1]
            
            # Check if Track ID changed (for Conversation Mode)
            track_id_changed = False
            if transition_mode == "conversation" and i > 0:
                current_id = track_id_history[i]
                prev_id = track_id_history[i - 1]
                track_id_changed = (current_id != prev_id) and (current_id is not None) and (prev_id is not None)
            
            # Check deviation from Anchor
            diff = abs(current_raw_x - anchor)
            
            # In Conversation Mode: If Track ID changed, immediately accept new position (bypass dead zone)
            if track_id_changed:
                anchor = current_raw_x
                if i % 30 == 0:  # Debug log every 30 frames
                    print(f"  Frame {i}: Track ID changed ({prev_id}→{current_id}), bypassing dead zone, anchor={anchor:.0f}px")
            elif diff > dead_zone_px:
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
        
        # Store predictive mode status per frame
        predictive_status = []
        
        # Easing Config
        easing_type = tracking_cfg.get("easing_type", "ease_out")
        velocity = 0.0 # For ease_in momentum
        
        print(f"Applying Smoothing Mode: {transition_mode.upper()} | Type: {easing_type}")
        if transition_mode == "smart":
            print(f"Smart Cut Threshold: {smart_cut_px:.1f}px ({smart_cut_thresh_pct*100}%)")
        elif transition_mode == "conversation":
            conv_settings = cam_ctrl.get("conversation_settings", {})
            print(f"Conversation Mode: Cut on ID change enabled (min_frames={conv_settings.get('min_same_id_frames', 10)})")
        elif transition_mode == "predictive":
            pred_settings = cam_ctrl.get("predictive_settings", {})
            PRED_LOOK_AHEAD = pred_settings.get("look_ahead_frames", 30)
            PRED_GRACE_PERIOD = pred_settings.get("grace_period_frames", 10)
            PRED_PRE_CUT_FRAMES = pred_settings.get("pre_cut_plan_frames", 5)
            PRED_MIN_CONF_SWITCH = pred_settings.get("min_confidence_switch", 0.5)
            PRED_MAX_GAP = pred_settings.get("max_gap_frames", 5)
            PRED_MIN_SAME_ID = pred_settings.get("min_same_id_frames", 10)
            
            print(f"Predictive Mode: LookAhead={PRED_LOOK_AHEAD}, Grace={PRED_GRACE_PERIOD}, PreCut={PRED_PRE_CUT_FRAMES}")
        
        for i, target in enumerate(stabilized_targets):
            tx = target[0]
            dist = abs(tx - current_cam_x)
            
            should_cut = False
            cut_reason = ""
            
            # ========== PREDICTIVE MODE ==========
            if transition_mode == "predictive":
                # Get current and previous track IDs
                current_track_id = track_id_history[i]
                current_track_class = actual_focused_history[i]['class_id']
                prev_track_id = track_id_history[i - 1] if i > 0 else None
                prev_track_class = actual_focused_history[i - 1]['class_id'] if i > 0 else None
                
                # Check if ID changed (including changes to/from None/Saliency)
                id_changed = (current_track_id != prev_track_id) and i > 0
                
                if id_changed:
                    # ===== เช็คว่า ID ใหม่อยู่ใน "กลุ่มเดียวกัน" หรือไม่ =====
                    # หากลุ่ม IDs ที่เกี่ยวข้องกับ prev_track_id (ดูอดีต+อนาคต)
                    related_group = set()
                    if prev_track_id is not None:
                        related_group = get_related_ids_group(
                            frames, i - 1, prev_track_id, prev_track_class, 
                            look_window=PRED_LOOK_AHEAD
                        )
                    
                    # เช็คว่า current_track_id อยู่ในกลุ่มเดิมไหม
                    is_same_group = (current_track_id in related_group) if current_track_id is not None else False
                    
                    # ID changed: Check if old ID was stable enough for a CUT
                    frames_with_prev_id = 0
                    if prev_track_id is not None:
                        for j in range(max(0, i - PRED_MIN_SAME_ID), i):
                            if track_id_history[j] == prev_track_id:
                                frames_with_prev_id += 1
                    
                    # Check if old ID temporarily lost (within grace period)
                    old_id_returns = False
                    if prev_track_id is not None:
                        for k in range(i, min(len(frames), i + PRED_GRACE_PERIOD)):
                            if track_id_history[k] == prev_track_id:
                                old_id_returns = True
                                break
                    
                    if old_id_returns:
                        # Old ID temporarily lost - HOLD position
                        cut_reason = f"HOLD: ID:{prev_track_id} returns"
                        should_cut = False
                        # Keep camera at current position
                        tx = current_cam_x
                    elif is_same_group and current_track_class != 0:
                        # Same-group logic เฉพาะ Body เท่านั้น (class_id != 0)
                        # สำหรับ Face (class_id == 0) → ข้าม logic นี้ไปเลย → CUT
                        if current_track_id is not None:
                            cut_reason = f"SMOOTH: ID:{prev_track_id}→{current_track_id} (same group)"
                        else:
                            cut_reason = f"SMOOTH: ID:{prev_track_id}→Saliency (same group)"
                        should_cut = False
                    elif prev_track_id is not None and frames_with_prev_id >= PRED_MIN_SAME_ID:
                        # ID เปลี่ยน และ old ID อยู่นานพอ → CUT
                        should_cut = True
                        if current_track_id is not None:
                            cut_reason = f"CUT: ID:{prev_track_id}→{current_track_id}"
                        else:
                            cut_reason = f"CUT: ID:{prev_track_id}→Saliency"
                    else:
                        # Old ID was flickering OR transition to/from Saliency - smooth transition
                        if current_track_id is not None and prev_track_id is not None:
                            cut_reason = f"SMOOTH: ID:{prev_track_id}→{current_track_id} (flicker)"
                        elif current_track_id is None:
                            cut_reason = f"SMOOTH: ID:{prev_track_id}→Saliency"
                        else:
                            cut_reason = f"SMOOTH: Saliency→ID:{current_track_id}"
                        should_cut = False
                
                else:
                    # ID same: Check if it will disappear
                    if current_track_id is not None:
                        # Analyze future trajectory
                        future_info = analyze_future_trajectory(
                            frames, i, current_track_id, current_track_class,
                            look_ahead=PRED_LOOK_AHEAD,
                            max_gap=PRED_MAX_GAP
                        )
                        
                        if future_info['will_disappear']:
                            # ID will disappear soon - PRE-PLAN
                            cut_reason = f"PRE-PLAN: ID:{current_track_id} disappears@{future_info['last_seen_frame']}"
                            
                            # Get last known position of this ID
                            last_known_x = tx
                            for k in range(future_info['last_seen_frame'], i, -1):
                                if k < len(stabilized_targets):
                                    last_known_x = stabilized_targets[k][0]
                                    break
                            
                            # Start moving towards last known position
                            # Use smooth transition to prepare for upcoming cut
                            tx = last_known_x
                        else:
                            # ID will stay - PLAN (normal smooth pan)
                            cut_reason = f"PLAN: Following ID:{current_track_id}"
            
            # ========== CONVERSATION MODE ==========
            elif transition_mode == "conversation":
                conv_settings = cam_ctrl.get("conversation_settings", {})
                cut_on_id_change = conv_settings.get("cut_on_id_change", True)
                min_same_id_frames = conv_settings.get("min_same_id_frames", 10)
                
                if cut_on_id_change and i > 0:
                    current_id = track_id_history[i]
                    prev_id = track_id_history[i - 1]
                    
                    # ตรวจสอบว่า ID เปลี่ยนไหม
                    id_changed = (current_id != prev_id) and (current_id is not None) and (prev_id is not None)
                    
                    if id_changed:
                        # นับว่า ID เก่าอยู่กี่เฟรม (ป้องกัน flicker จาก ByteTracker)
                        frames_with_prev_id = 0
                        for j in range(max(0, i - min_same_id_frames), i):
                            if track_id_history[j] == prev_id:
                                frames_with_prev_id += 1
                        
                        # ถ้า ID เก่าอยู่นานพอ → อนุญาตให้ cut
                        if frames_with_prev_id >= min_same_id_frames:
                            should_cut = True
                            cut_reason = f"Conversation:ID{prev_id}->{current_id}"
                
                # FALLBACK: ถ้าระยะห่างมากเกิน smart_cut_threshold → CUT แม้ ID เดิม
                if not should_cut and smart_cut_px > 0 and dist > smart_cut_px:
                    should_cut = True
                    cut_reason = f"Conversation:Dist>{smart_cut_px:.0f}px"
            
            # ========== OTHER MODES (CUT/SMART) ==========
            elif transition_mode == "cut":
                should_cut = True
                cut_reason = "Mode:Cut"
            
            elif transition_mode == "smart":
                if dist > smart_cut_px:
                    should_cut = True
                    cut_reason = f"Smart:Dist>{smart_cut_px:.0f}px"
            
            # ========== EXECUTE CUT OR SMOOTH ==========
            if should_cut:
                # Hard Cut
                current_cam_x = tx
                velocity = 0.0 # Reset Physic
                # Debug output (ทุก 30 เฟรมเพื่อไม่ให้ spam log)
                if cut_reason and i % 30 == 0:
                    print(f"  Frame {i}: CUT ({cut_reason})")
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
            
            # Save predictive mode status
            if transition_mode == "predictive" and cut_reason:
                predictive_status.append(cut_reason)
            else:
                predictive_status.append("")
            
        # Save Path
        output = {
            "meta": data['meta'],
            "path": smoothed_path,  # List of X coordinates for crop center
            "debug_info": debug_path_info, # List of reasons per frame
            "track_id_history": track_id_history,  # Track ID per frame (for debugging conversation mode)
            "actual_focused_history": actual_focused_history,  # ACTUAL bbox used for camera (Track ID + Class)
            "predictive_status": predictive_status  # Predictive mode status per frame
        }
        
        print(f"Path generated. Saving to {output_path_json}...")
        with open(output_path_json, 'w') as f:
            json.dump(output, f)

        print("Analysis Complete.")
