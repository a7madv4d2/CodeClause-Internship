Real-Time Facial Recognition & Object Detection System

An advanced computer vision pipeline for high-precision face detection and identification.

</div>

📖 Overview

This project implements an end-to-end Computer Vision system designed for real-time detection in complex environments. By leveraging the YOLOv8 architecture, the system moves beyond generic object detection to offer a specialized Facial Recognition module capable of identifying distinct individuals with high confidence.

Wrapped in a robust Streamlit interface, the application democratizes access to advanced AI, enabling non-technical users to deploy security, attendance tracking, or demographic analytics solutions effortlessly.

🚀 Key Features

🎭 Dual-Model Architecture Seamlessly toggle between the pre-trained COCO model (80 generic classes) and a custom-trained Facial Recognition network without restarting the application.

⚡ Real-Time Inference Optimized for high-FPS performance on standard hardware, utilizing YOLOv8's efficient backend for low-latency video stream processing.

🎛️ Interactive Dashboard A professional-grade UI built with Streamlit, featuring dynamic controls for confidence thresholds, NMS (Non-Maximum Suppression), and real-time visualization layers.

🧠 Transfer Learning Pipeline Includes modular scripts for fine-tuning the model on custom datasets, facilitating rapid adaptation for specific individuals, emotion analysis, or mask compliance verification.

🛠️ Tech Stack

Component

Technology

Description

Core AI

Ultralytics YOLOv8

State-of-the-art Deep Learning object detection model.

Computer Vision

OpenCV & PIL

High-performance image processing and video stream handling.

Frontend

Streamlit

Rapid deployment framework for data science web applications.

Language

Python 3.x

Core programming language for logic and integration.

💻 Installation

Clone the repository

git clone [https://github.com/yourusername/facial-recognition-system.git](https://github.com/yourusername/facial-recognition-system.git)
cd facial-recognition-system



Install Dependencies

pip install ultralytics streamlit opencv-python-headless pillow



Run the Application

streamlit run app.py



🧠 How to Train for Specific Faces

To customize this system to recognize specific people (or use your downloaded dataset):

Prepare Data: Ensure your downloaded facial recognition dataset is organized with images of faces.

If using your own photos: Collect 50+ images per person/face you want to identify.

Label Data: Upload your images to Roboflow.

Label the faces with the person's name (e.g., "John_Doe", "Jane_Smith").

Export: Export the dataset in YOLOv8 format.

Update Config: Update the data.yaml path in train_custom_model.py to point to your new facial dataset.
