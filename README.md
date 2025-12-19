# CodeClause-Internship
My projects from CodeClause Artificial Intelligence Internship.

👁️ Real-Time Facial Recognition & Object Detection System

📖 Overview

This project is an end-to-end Computer Vision system capable of detecting objects in images, videos, and real-time streams. It features a specialized Custom Trained Model optimized for Facial Recognition, allowing the system to detect and identify human faces with high precision, moving beyond simple generic object detection.

The system is wrapped in a user-friendly Streamlit Web Interface, making it accessible to non-technical users for security, attendance, or analytics applications.

🚀 Key Features
Dual-Model Architecture: Instantly switch between the standard COCO model (general objects) and the custom Facial Recognition model.

Real-Time Inference: High-FPS face detection on video streams using YOLOv8's optimized backend.

Interactive Dashboard: A browser-based UI built with Streamlit to adjust confidence thresholds and visualize detection results dynamically.

Transfer Learning: Scripts included to fine-tune the model on specific facial datasets (e.g., specific individuals, emotion detection, or mask compliance).

🛠️ Tech Stack
Core AI: Ultralytics YOLOv8 (Deep Learning)

Image Processing: OpenCV & PIL

Interface: Streamlit

Language: Python
