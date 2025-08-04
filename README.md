# SmartCare AI Light

SmartCare AI Light is an **AI-powered intelligent traffic light system** developed during the **SPIED 2025 International Workshop in China**.  
The system uses computer vision to detect **vulnerable pedestrians** — such as wheelchair or cane users — and **automatically extends the green light** until they have safely crossed.  
This not only improves **pedestrian safety** but also **optimizes traffic flow** for drivers.

---

## 📜 Background

This idea was inspired by a real-life experience:  
While helping a friend with a broken leg cross the street, we couldn’t make it before the light turned red.  
It raised the question: *Why don’t traffic lights give more time to those who really need it?*

SmartCare AI Light solves this problem by **detecting** and **responding** in real time.

This project was created by **Team OneAsia** (members from Korea, Japan, and China) during the **SPIED 2025 International Workshop held in China**.

---

## 🚀 Features

- **Real-time pedestrian detection** using CCTV/IP camera
- **Two AI-based methods**:
  1. **YOLO-based Object Detection** — Detects objects like wheelchairs, walking canes, or crutches
  2. **Pose-based Classification** — Uses human pose keypoints to detect mobility difficulty
- **Automatic traffic light control** to extend pedestrian green time
- **Traffic optimization**: Prevents wasted green light for drivers

---

## 🛠️ Technology Stack

- **Python 3**
- **OpenCV** — Camera video processing
- **YOLOv8 / YOLOv11-Pose** — Object detection & pose estimation
- **PyTorch** — Model training/inference
- **Raspberry Pi / Microcontroller** — Traffic light control (simulation or real-world)
- **Rule-based logic** or **ML classification** for decision-making

---

## 📂 Project Structure

