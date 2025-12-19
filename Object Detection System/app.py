import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from PIL import Image
import numpy as np

# Page Config
st.set_page_config(page_title="AI Object Detection System", layout="wide")

st.title("Object Detection System")
st.markdown("Compare the **Standard Model** vs. **Facial Recognition Model**")

# Sidebar - Settings
st.sidebar.header("Model Settings")

# Model Selection
model_type = st.sidebar.radio(
    "Select Model Strategy", 
    ["Standard (COCO - 80 Classes)", "Custom (Your Trained Model)"]
)

# Confidence Slider
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)

# Load the selected model
try:
    if model_type == "Standard (COCO - 80 Classes)":
        model = YOLO('yolov8n.pt')
        st.sidebar.success("Loaded Standard Model")
    else:
        # POINT THIS TO YOUR SAVED WEIGHTS
        # Usually: runs/detect/my_custom_model/weights/best.pt
        custom_path = 'runs/detect/my_custom_model/weights/best.pt'
        
        if os.path.exists(custom_path):
            model = YOLO(custom_path)
            st.sidebar.success("Loaded Custom Model")
        else:
            st.sidebar.error(f"Could not find model at {custom_path}")
            st.sidebar.warning("Falling back to Standard Model")
            model = YOLO('yolov8n.pt')
except Exception as e:
    st.error(f"Error loading model: {e}")

# Main Content - Input
source_type = st.radio("Select Source", ["Image Upload", "Video Upload"])

if source_type == "Image Upload":
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        # Convert to format YOLO understands
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)
        
        if st.button("Detect Objects"):
            with st.spinner("Analyzing..."):
                # Run Inference
                results = model.predict(image, conf=conf_threshold)
                
                # Plot results
                res_plotted = results[0].plot()
                
                # Show Result
                st.image(res_plotted, caption="Detected Objects", use_container_width=True)
                
                # Show Stats
                st.write(f"Detected {len(results[0].boxes)} objects.")

elif source_type == "Video Upload":
    uploaded_video = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi'])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        if st.button("Process Video (First 5 seconds)"):
            vf = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret:
                    break
                
                # Process Frame
                results = model.predict(frame, conf=conf_threshold)
                res_plotted = results[0].plot()
                
                # Display
                stframe.image(res_plotted, channels="BGR", use_container_width=True)
            
            vf.release()