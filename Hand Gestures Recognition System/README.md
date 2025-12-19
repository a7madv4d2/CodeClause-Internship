# ✋ Real-Time Hand Gesture Recognition System

A robust **Human–Computer Interaction (HCI)** system that uses **Deep Learning** and **Computer Vision** to interpret hand gestures in real time.  
The project features a **custom VGG-style Convolutional Neural Network (CNN)** and a professional **Heads-Up Display (HUD)** interface for controlling virtual environments or devices without physical touch.

---

## 🚀 Key Features

### 🧠 Deep Learning Brain
- Custom **VGG-style CNN architecture**
- Includes **Batch Normalization** and **Dropout**
- Robust classification of **10 distinct hand gestures**

### ⚡ Real-Time Inference
- Low-latency prediction pipeline
- **OpenCV** for frame processing
- **TensorFlow** for real-time inference

### 🎛️ Professional HUD Interface
- 📊 **Live Probability Analytics**  
  Sidebar visualization showing confidence scores for all gesture classes
- 🎯 **Dynamic ROI**  
  Bounding box color changes based on prediction confidence
- 🎚️ **Sensitivity Tuner**  
  Real-time slider to adjust binary thresholding and handle lighting variations

### 🖐️ Functional Gesture Mapping
- 10 gestures mapped to simulated system commands  
  *(e.g., volume control, media navigation)*

---

## 🛠️ Tech Stack

- **Language:** Python
- **Computer Vision:** OpenCV (`cv2`)
- **Deep Learning:** TensorFlow / Keras
- **Data Manipulation:** NumPy, Shutil :contentReference[oaicite:1]{index=1}

---

## 📂 Dataset Source

This project was trained on the **LeapGestRecog – Hand Gesture Recognition Database**.

- **Source:** Kaggle – Hand Gesture Recognition Database
- **Content:**  
  20,000 Infrared (IR) images  
  10 gestures × 10 subjects

**Important Note:**  
Although the dataset contains **IR images**, this project applies **data augmentation** and **adaptive thresholding** to generalize effectively to standard **RGB webcam feeds**. 
---

## 🎮 Gesture Controls

The system recognizes the following gestures and maps them to actions:

| Gesture | Action | Symbol |
|------|-------|-------|
| Palm | Pause Video | ✋ |
| Fist | Grab / Hold | ✊ |
| Thumb Up | Volume Up / Like | 👍 |
| Thumb Down | Volume Down / Dislike | 👎 |
| Index Point | Click / Select | ☝️ |
| OK Sign | Confirm Selection | 👌 |
| Palm Moved | Swipe Screen | 👋 |
| Fist Moved | Drag Object | ✊ |
| C-Sign | Copy | 🇨 |
| L-Sign | Lock Screen | 🇱 |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/a7madv4d2/Hand-Gestures-Recognition-System.git
cd Hand-Gestures-Recognition-System
```

### 2️⃣ Install Dependencies
```
pip install opencv-python tensorflow numpy matplotlib
```

## ▶️ Running the Pipeline

The project follows a **Jupyter Notebook–based workflow**:

### 1️⃣ Organize Data
Run the data organization script to flatten the dataset structure into clean class folders.

### 2️⃣ Train Model
Run the training notebook or script to generate the trained model file: gesture_model_robust.h5

### 3️⃣ Run Application
Execute the application script to launch the webcam-based hand gesture recognition interface.



