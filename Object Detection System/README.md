# 🎯 Real-Time Facial Recognition & Object Detection System

An advanced computer vision pipeline for **high-precision face detection and identification**, built on **YOLOv8** and deployed through an **interactive Streamlit dashboard**.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Installation]

---

## 📖 Overview

This project implements an **end-to-end computer vision system** designed for real-time object detection and facial recognition.

Using **YOLOv8**, the system supports both generic object detection and **identity-based facial recognition** through custom-trained models. A **Streamlit web interface** makes the solution accessible to non-technical users.

---

## 🚀 Key Features

- 🎭 **Dual-Model Architecture**  
  Switch between COCO object detection and custom facial recognition models.

- ⚡ **Real-Time Inference**  
  Optimized for low-latency, high-FPS video processing.

- 🎛️ **Interactive Dashboard**  
  Live confidence threshold and NMS controls using Streamlit.

- 🧠 **Transfer Learning Pipeline**  
  Fine-tune models for specific faces or custom datasets.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|--------|-----------|------------|
| Core AI | YOLOv8 | State-of-the-art object detection |
| Vision | OpenCV, PIL | Image and video processing |
| UI | Streamlit | Interactive dashboard |
| Language | Python 3 | Core programming language |

---
## ⚙️ Installation & Setup (Object Detection System)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/a7madv4d2/CodeClause-Internship.git
cd CodeClause-Internship/Object\ Detection\ System
```
### 2️⃣ Install Dependencies

```
pip install ultralytics streamlit opencv-python-headless pillow
```
