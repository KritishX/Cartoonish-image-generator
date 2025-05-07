# app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(page_title="Cartoonify Your Image", page_icon="🎨", layout="centered")

# --- Custom CSS Styling (Dark Minimal) ---
custom_css = """
<style>
    /* Global Background */
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        padding: 20px;
    }

    /* Title Styling */
    h1 {
        text-align: center;
        color: #ffffff;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }

    /* Subtitle Text */
    .stMarkdown > div {
        text-align: center;
        font-size: 16px;
        color: #B0B0B0;
    }

    /* File Uploader Styling */
    .stFileUploader {
        background-color: #1e1e1e;
        padding: 14px;
        border-radius: 12px;
        border: 1px dashed #333333;
        margin-top: 25px;
        margin-bottom: 30px;
    }

    /* Buttons Styling */
    .stButton button, .stDownloadButton button {
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        transition: all 0.3s ease-in-out;
        background-color: #333333;
        color: #ffffff;
        border: none;
    }

    .stButton button:hover, .stDownloadButton button:hover {
        background-color: #444444;
        transform: translateY(-1px);
    }

    /* Image Columns Styling */
    .element-container {
        padding: 10px;
    }

    /* Image Subheaders */
    h3 {
        text-align: center;
        color: #E0E0E0;
        font-weight: 500;
        margin-top: 10px;
    }

    /* Image Display */
    .stImage > img {
        border-radius: 12px;
        box-shadow: 0px 0px 15px rgba(0,0,0,0.7);
        margin-top: 10px;
    }

    /* Download button spacing */
    .stDownloadButton {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- App Title ---
st.title("🎨 Cartoonify Your Image")

st.write("Upload an image below and see it transform into a **cartoon-style artwork** — simple & elegant.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload an image (JPG/JPEG/PNG)", type=["jpg", "jpeg", "png"])

# --- Cartoonify Function ---
def cartoonify(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Bilateral Filter
    img_color = img_rgb
    for _ in range(7):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=75, sigmaSpace=75)

    # Grayscale + Blur
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)

    # Edge Detection
    edges = cv2.adaptiveThreshold(img_blur, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY,
                                   blockSize=9,
                                   C=2)

    # Combine edges + color
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cartoon = cv2.bitwise_and(img_color, edges_colored)

    return cartoon

# --- Process Image ---
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    if image_np.shape[-1] == 4:  # Remove alpha if present
        image_np = image_np[..., :3]

    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    cartoon = cartoonify(image_cv)

    # --- Display Side-by-Side Images ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Original")
        st.image(image, use_column_width=True)

    with col2:
        st.markdown("### Cartoonified")
        st.image(cartoon, use_column_width=True)

    # --- Prepare image for download ---
    result_pil = Image.fromarray(cartoon)
    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Cartoonified Image",
        data=byte_im,
        file_name="cartoonified_image.jpg",
        mime="image/jpeg"
    )
