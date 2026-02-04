# 🎬 Debug View Feature

## ✅ คุณสมบัติ

เมื่อรันด้วย `--debug-view` จะแสดง:

1. **Track ID Labels** บน bounding box ทุกตัว
2. **เส้นขอบเขต Cut/Plan threshold** (เส้นสีเหลือง)
3. **Camera center line** (เส้นสีส้ม)
4. **Mode และ Decision info** (มุมซ้ายบน)

---

## 🎨 Visual Layout

```
┌────────────────────────────────────────────────────────┐
│ Mode: CONVERSATION          │                          │
│ Cut Threshold: 768px (40%)  │                          │
│ Decision: Face ID:5                                    │
│                             │                          │
│                  │          │                          │
│    ID:5          │          │                          │
│  ┌──────┐        │        ┌──────┐                     │
│  │ Face │        │        │ ID:8 │                     │
│  └──────┘        │        └──────┘                     │
│                  │                                      │
│      ▲           ▲           ▲                          │
│      │           │           │                          │
│   Yellow      Orange      Yellow                        │
│   (Left       (Camera     (Right                        │
│  Threshold)    Center)   Threshold)                     │
└────────────────────────────────────────────────────────┘
```

---

## 🎯 การใช้งาน

### รัน Debug View:
```bash
python auto_reframe.py video.mp4 --output output.mp4 --debug-view
```

### Output Video จะแสดง:
- ✅ Track ID labels บนทุก bbox
- ✅ Yellow threshold lines (ขอบเขต cut)
- ✅ Orange camera center line
- ✅ Mode และ decision info
- ✅ Magenta crop box

---

## 🎨 สีและความหมาย

| สี | Component | ความหมาย |
|----|-----------|---------|
| 🟢 Green | Face bbox | ใบหน้า (Class 0) |
| 🔵 Blue | Body bbox | ตัวคน (Class 1) |
| 🟡 Yellow | Threshold lines | ขอบเขต cut/plan |
| 🟠 Orange | Camera center | จุดศูนย์กล้อง |
| 🟣 Magenta | Crop box | พื้นที่ที่ crop |
| ⚪ White | Text labels | ข้อมูลต่างๆ |

---

## 🔍 การอ่าน Debug Info

### ข้อมูลที่แสดง:
- **Mode**: SMART หรือ CONVERSATION
- **Decision Reason** เช่น:
  - `Face ID:4` - กำลัง Lock หน้าคน ID 4
  - `Face ID:4 (Future)` - Pre-position ไปหา Face อนาคต
  - `Continuity ID:1→4` - สลับ Face โดยไม่กระโดดกลับ
  - `Body ID:2` - กำลัง Lock Body
  - `Saliency` - ใช้ Saliency point

---

## 🆕 Features ใหม่ล่าสุด (Feb 2026)

### 1. Face Look-Ahead Logic
- ระบบมองอนาคตเพื่อ Pre-position กล้องไปหา Face ก่อนที่จะปรากฏ
- ลด Flicker และ Snap เมื่อเปลี่ยน Face

### 2. Position Continuity Check
- ป้องกันกล้องกระโดดกลับไปตำแหน่งเดิมเมื่อ Face ใหม่ปรากฏ
- ทำให้ Transition ราบรื่นมากขึ้น

### 3. Confidence-Based Switching
- สลับจาก Ghost Track ไป Face จริงทันทีเมื่อ Confidence ต่างกันมาก

---

**Updated:** 2026-02-04
