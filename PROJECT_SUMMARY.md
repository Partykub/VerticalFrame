# 📄 Auto-Reframe AI Pipeline: Project Summary

**Updated:** 2026-01-29
**Status:** ✅ Production Ready 2.0 (High-Fidelity)

---

## 1. 🎯 Project Objective
สร้างระบบอัตโนมัติด้วย AI เพื่อแปลงวิดีโอแนวนอน (Horizontal 16:9) ให้เป็นวิดีโอแนวตั้ง (Vertical 9:16) สำหรับ Social Media (TikTok, Reels, Shorts) โดยมุ่งเน้น **คุณภาพสูงสุด (Visually Lossless)** เสมือนมีผู้กำกับและ Editor มืออาชีพมานั่งทำทีละเฟรม

---

## 2. 🏗️ System Architecture (The "Lossless" Pipeline)
ระบบทำงานแบบ **3-Stage Offline Process** ที่เน้นคุณภาพข้อมูลในทุกจุดเชื่อมต่อ (Data Integrity)

### 🔄 Phase 1: Scanner (The Eyes)
*   Scanning แบบละเอียด (Frame-by-Frame) โดยไม่มีการข้ามการตรวจจับ
*   **AI Stack:** Hybrid 3 Layers
    1.  **YOLOv8:** ตรวจจับคน (Body) และสิ่งของ
    2.  **MediaPipe:** ตรวจจับใบหน้า (Face) เพื่อความแม่นยำระดับรูขุมขน
    3.  **Saliency Map:** ตรวจจับ "จุดที่น่าสนใจ" (Visual Saliency) เมื่อไม่มีคนในฉาก
*   **Output:** `temp_tracking_data.json`

### 🧠 Phase 2: Analyzer (The Brain)
*   **Look-Ahead Buffer:** ระบบอ่านอนาคตล่วงหน้า 60 เฟรม+ เพื่อตัดสินใจว่าจะตัดกล้องเมื่อไหร่ หรือจะแพนกล้องตามใคร
*   **Smoothing Engine:** ใช้คณิตศาสตร์ `Sine-In-Out Easing` เพื่อจำลองการหมุนเลนส์กล้องภาพยนตร์ (เริ่มช้า-เร่ง-จบช้า) ไม่ใช่ Linear Robot Movement
*   **Output:** `temp_camera_path.json`

### 🎨 Phase 3: High-Fidelity Renderer (The Heart of Quality) 🌟
**Major Upgrade (2026-01-29):** เปลี่ยนจาก OpenCV Writing เป็น **Direct FFmpeg Piping**
*   **Process:**
    1.  `OpenCV` อ่าน Raw Pixel Matrix (BGR) จาก Source ลง RAM
    2.  ส่งข้อมูลดิบผ่าน `stdin pipe` เข้าสู่ FFmpeg Process โดยตรง (ไม่มีการ Save รูปเป็นไฟล์ JPG ระหว่างทาง)
    3.  FFmpeg บีบอัดด้วย **libx264 (CRF 18)** และ **Preset Slow**
*   **Result:**
    *   **Bitrate เพิ่มขึ้น ~3 เท่า** เพื่อรักษา Detail ขอบภาพหลังการ Crop
    *   **Visually Lossless:** ตาเปล่าแยกไม่ออกระหว่าง Source กับ Output
    *   **Integrity Check:** ระบบนับเฟรม Input vs Output ตอนจบงาน เพื่อยืนยันว่าไม่มีเฟรมหาย (Frame Drop = 0)

---

## 3. 💡 Core Intelligence Features

### 3.1 Smart Priority System
Director AI จะเลือกเป้าหมายตามลำดับความสำคัญ (Hierarchy):
1.  **Face (ใบหน้า):** พระเอกตัวจริง (ถ้าเห็นหน้า ให้ Lock หน้าทันที)
2.  **Body (ตัวคน):** ถ้าเห็นแค่ตัว ให้ Lock ช่วงอก (Upper Body)
3.  **Saliency (Context):** ถ้าไม่มีคน ให้มองสิ่งที่เด่นที่สุดในฉาก (เช่น อาหาร, แมว, วิว)

### 3.2 Dynamic Argumentation
*   **--debug-view:** โหมดพิเศษสำหรับ Dev/QC จะ Render วิดีโอแบบ 2 จอ (ซ้าย: ต้นฉบับ+Overlay, ขวา: ผลลัพธ์) เพื่อให้ตรวจสอบได้ว่า AI คิดอะไรอยู่
*   **Auto-Fallback:** ถ้า User ไม่สั่งอะไร จะ Render แบบ Clean Vertical (9:16) เพื่อนำไปใช้งานทันที

---

## 4. 🛠️ Tech Stack & Dependencies

See also **[DEPLOY.md](DEPLOY.md)** for Docker deployment and **[README.md](README.md)** for install steps.

### Core Engine
*   **Python:** 3.8+
*   **FFmpeg (Binary):** หัวใจหลักของการจัดการ Video/Audio (ใช้ผ่าน `subprocess` pipe)

### AI & Vision
*   `ultralytics` (YOLOv8)
*   `mediapipe` (Face Mesh)
*   `opencv-contrib-python` (Saliency — **not** `opencv-python`)
*   `supervision` (ByteTracker Algorithm)
*   `deep-sort-realtime` (DeepSORT tracking)
*   `OpenAI CLIP` (Re-ID embedder for DeepSORT, default `clip_RN50`)

### Mathematics & Logic
*   `numpy` (Matrix Operations)
*   `scipy` (Signal Processing / Smoothing)

---

## 5. 📊 Quality Assurance Metrics

| Parameter | Value | Meaning |
| :--- | :--- | :--- |
| **Pipeline Mode** | Offline | ประมวลผลล่วงหน้าทั้งไฟล์ (Non-Realtime) เพื่อความแม่นยำ |
| **Video Codec** | H.264 (libx264) | มาตรฐานสากล รองรับทุก Platform |
| **Compression** | **CRF 18** | High Quality (Visually Lossless) |
| **Audio Codec** | AAC | High Fidelity Sound |
| **Frame Drop** | 0.00% | รับประกันด้วย Integrity Check System |

---
*Created by Antigravity Team*
