# 📏 Camera Transition Modes & Threshold

## 🎯 Transition Modes

### 1. Smart Mode
```json
"transition_mode": "smart",
"smart_cut_threshold_percent": 0.4
```

**พฤติกรรม:**
- ถ้าระยะทางเป้าหมาย < 40% ของ frame width → **Smooth Pan**
- ถ้าระยะทางเป้าหมาย ≥ 40% → **Hard Cut**

### 2. Conversation Mode
```json
"transition_mode": "conversation"
```

**พฤติกรรม:**
- ถ้าเปลี่ยน Track ID → **Hard Cut** (ไม่สนใจระยะทาง)
- ถ้า ID เดิม + อยู่ในเขต → **Smooth Pan**

---

## 📐 Threshold Visualization (Debug View)

### เส้นต่างๆ และความหมาย:

| เส้น | สี | ความหมาย |
|-----|-----|---------|
| Threshold Lines | 🟡 Yellow | ขอบเขตการตัดสินใจ Cut vs Pan |
| Camera Center | 🟠 Orange | จุดศูนย์กลางของกล้อง (crop center) |
| Crop Box | 🟣 Magenta | พื้นที่ที่จะถูก crop |

### กฎพื้นฐาน:
- เป้าหมายอยู่ **ระหว่างเส้นเหลือง** → **SMOOTH PAN** ✓
- เป้าหมายอยู่ **นอกเส้นเหลือง** → **HARD CUT** ✂️

---

## 🔧 Smart Lock System

### Config Parameters (`config.json`):

```json
"smart_lock": {
    "look_ahead_frames": 45,
    "switch_threshold_ratio": 0.75,
    "grace_period_frames": 45
}
```

| Parameter | ค่า | ความหมาย |
|-----------|-----|----------|
| `look_ahead_frames` | 45 | ดูอนาคตกี่เฟรมก่อนเปลี่ยนเป้า |
| `switch_threshold_ratio` | 0.75 | ต้องชนะกี่ % ถึงเปลี่ยนคน |
| `grace_period_frames` | 45 | รอคนเก่านานแค่ไหนถ้าหาย |

---

## 🆕 Director Configuration

### Face Look-Ahead (Skip Body):

```json
"director": {
    "face_priority_size_weight": 1.0,
    "body_offset_y": 0.3,
    "body_to_face_lookahead": 15,
    "body_face_overlap_threshold": 0.3,
    "face_switch_threshold": 1.05
}
```

| Parameter | ค่า | ความหมาย |
|-----------|-----|----------|
| `body_to_face_lookahead` | 15 | กี่เฟรมที่ดูอนาคตหา Face |
| `body_face_overlap_threshold` | 0.3 | % overlap ที่ถือว่าคนเดียวกัน |
| `face_switch_threshold` | 1.05 | Face ใหม่ต้องใหญ่กว่ากี่เท่า |

---

## 📊 Console Log Messages

### Smart Mode:
```
Frame X: CUT (Smart:Dist>768px)
```

### Conversation Mode:
```
Frame X: CUT (Conversation:ID144->141)
```

### Face Look-Ahead:
```
[Director] Skip Body, use future Face #4
```

### Position Continuity:
```
Continuity ID:1→4 (dist:50<200)
```

---

## 🧪 ทดสอบ Feature

```bash
# รัน debug view
python auto_reframe.py video.mp4 --output test.mp4 --debug-view

# ดู console output:
# - "[Director] Skip Body, use future Face #X" เมื่อ skip body
# - "Frame X: CUT (...)" เมื่อมีการ cut
```

---

**Updated:** 2026-02-04
