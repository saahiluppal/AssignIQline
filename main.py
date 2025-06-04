import streamlit as st
import os
import random
from PIL import Image
import time

# Page configuration
st.set_page_config(page_title="Rapid Test Line Detection", layout="centered")
st.title("📷 Rapid Test Line Detection")

# Image selection section
st.header("Select a Random Image From the Database or Upload your own")

# Random selection button
random_image = None
uploaded_image = None
selected_image = None
if st.button("Select Random Image"):
    random_dir = "random_images/"
    if os.path.exists(random_dir):
        image_files = [f for f in os.listdir(random_dir) if f.lower().endswith(("jpg", "jpeg", "png"))]
        if image_files:
            random_file = random.choice(image_files)
            random_image = Image.open(os.path.join(random_dir, random_file))
            selected_image = random_image
            st.image(selected_image.resize((512, 512)), caption="Randomly Selected Image")
        else:
            st.warning("No valid images found in random_images/ directory.")
    else:
        st.error("random_images/ directory not found.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image.resize((512, 512)), caption="Uploaded Image", use_container_width=True)

    selected_image = uploaded_image

## 
import os
import cv2

import torch
from torchvision.ops import nms

from scipy.signal import find_peaks
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO

import tempfile

temp_dir = tempfile.mkdtemp()
##

def crop_from_xywh(np_img, xywh):
    x_center, y_center, width, height = xywh

    # Convert to xyxy
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    # Clip to image bounds
    h, w = np_img.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    return np_img[y1:y2, x1:x2]

def apply_nms(roi_results):
    
    if len(roi_results) == 0 or len(roi_results[0].boxes.xywh) == 0:
        print("No ROI found! Can't infer.")
    else:
        print(f"Found: {len(roi_results[0].boxes.xywh)} regions before NMS")
    
        boxes_xywh = roi_results[0].boxes.xywh.cpu()
        confidences = roi_results[0].boxes.conf.cpu()
        boxes_xyxy = torch.zeros_like(boxes_xywh)
    
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    
        iou_threshold = 0.4
        keep_indices = nms(boxes_xyxy, confidences, iou_threshold)
    
        print(f"Remaining after NMS: {len(keep_indices)} boxes")
    
        image = cv2.imread(image_path)
    
        for idx in keep_indices:
            x1, y1, x2, y2 = boxes_xyxy[idx].int().tolist()
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return image
##

def process_and_display_results(image: Image.Image):
    start = time.perf_counter()
    image.save(os.path.join(temp_dir, "source.jpg"))
    
    # Define Model
    ROI_model = YOLO('train8/weights/best.pt')
    LL_model = YOLO('train11/weights/best.pt')
    roi_results = ROI_model(os.path.join(temp_dir, "source.jpg"))
    
    if len(roi_results) == 0:
        print("No ROI found!, Can't infer")
    
    # apply_nms(roi_results)
    print(f"Found: {len(roi_results[0].boxes.xywh)} regions")
    
    for roi in roi_results:
        path = roi.path
        image = cv2.imread(os.path.join(temp_dir, "source.jpg"))
    
        for idx in range(len(roi.boxes.xywh)):
            # Contour Detection 
            image = image[30:-30].copy()
    
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            contrast_img = clahe.apply(gray)
            
            _, binary = cv2.threshold(contrast_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            lines = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / float(h)
                if aspect > 3 and 10 < h < 60 and w > 50:  # adjust these
                    lines.append((x, y, w, h))
    
            # Peak Signal Detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 30, 100)
            
            # Use horizontal projection
            projection = np.sum(edges, axis=1)
            peaks = np.where(projection > np.percentile(projection, 90))[0]
            
            # Group peaks (nearby rows are same line)
            peak, line_indices = find_peaks(projection, distance=50, height=np.percentile(projection, 90))
    
            
            xywh = roi.boxes.xywh.cpu().numpy()[idx].tolist()
            sub_img = crop_from_xywh(image, xywh)
            # sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
            sub_img_pil = Image.fromarray(sub_img)
            line_results = LL_model(sub_img_pil)

            end = time.perf_counter()
            elapsed = end - start
            st.write(f"Elapsed time: {elapsed:.4f} seconds")
            
            num_lines = len(line_results[0].boxes.xywh)
            st.write(f"Num of Lines Found: {len(line_results[0].boxes.xywh)}")        
    
            if num_lines == 0:
                st.markdown("## Invalid Test Results!")
            elif num_lines == 1:
                st.markdown("## Negative Test Results!")
            elif num_lines == 2:
                st.markdown("## Positive Test Results!")
            else:
                st.markdown("## Invalid Test Results!**")

            cols = st.columns(3)
            with cols[0]:
                st.image(cv2.resize(image, (480, 480)), caption="Original Image")

            with cols[1]:
                st.image(cv2.resize(sub_img, (480, 480)), caption="Cropped Region of Interest")

            with cols[2]:
                st.image(cv2.resize(line_results[0].plot(), (480, 480)), caption="Lines Found")
##

# Empty function to process image

# If an image is selected, process it
if selected_image or uploaded_image:
    process_and_display_results(selected_image)
