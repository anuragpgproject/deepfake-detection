"""
app.py

Streamlit web application for the Hybrid CNN + ViT Deepfake Detection
System. Lets users upload an image or video, runs the trained model,
and displays the predicted class, confidence score, and Grad-CAM /
attention explainability visualizations.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import cv2
import numpy as np
import streamlit as st

import config
from src.inference import load_model, predict_image, predict_video

st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🕵️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("🕵️ Deepfake Detector")
st.sidebar.markdown(
    """
This tool uses a **hybrid CNN + Vision Transformer** model to classify
uploaded images or videos as **Real** or **Fake**, with Grad-CAM and
attention-based explanations for transparency.

**Pipeline**
1. Face detection & cropping (MTCNN)
2. CNN branch — local artifact detection
3. ViT branch — global context modeling
4. Fusion + classification
5. Explainable AI overlays
"""
)
st.sidebar.markdown("---")
checkpoint_status = "✅ Trained weights found" if os.path.exists(config.BEST_MODEL_PATH) else "⚠️ No trained checkpoint found — run `python -m src.train` first"
st.sidebar.caption(checkpoint_status)
st.sidebar.caption(f"Device: `{config.DEVICE}`")

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("AI-Powered Deepfake Detection")
st.markdown(
    "Upload a face image or a short video clip to check whether it is "
    "**real** or **AI-generated / manipulated**."
)

tab_image, tab_video, tab_about = st.tabs(["📷 Image Detection", "🎞️ Video Detection", "ℹ️ About"])

model = load_model()

# --------------------------------------------------------------------------
# Image tab
# --------------------------------------------------------------------------
with tab_image:
    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"], key="image_uploader"
    )

    if uploaded_image is not None:
        file_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Analyzing image..."):
            result = predict_image(image_bgr, model=model)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(result["face_crop"], caption="Detected Face", use_container_width=True)
        with col2:
            st.image(result["gradcam_overlay"], caption="Grad-CAM (CNN attention)", use_container_width=True)
        with col3:
            st.image(result["attention_overlay"], caption="ViT Attention Map", use_container_width=True)

        st.markdown("---")
        label_color = "🟢" if result["label"] == "Real" else "🔴"
        st.subheader(f"{label_color} Prediction: **{result['label']}**")
        st.metric("Confidence", f"{result['confidence']}%")

        prob_col1, prob_col2 = st.columns(2)
        prob_col1.progress(int(result["probabilities"]["Real"]), text=f"Real: {result['probabilities']['Real']}%")
        prob_col2.progress(int(result["probabilities"]["Fake"]), text=f"Fake: {result['probabilities']['Fake']}%")

# --------------------------------------------------------------------------
# Video tab
# --------------------------------------------------------------------------
with tab_video:
    uploaded_video = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"], key="video_uploader"
    )
    num_frames = st.slider("Number of frames to sample", min_value=5, max_value=30, value=10)

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_video.read())
            tmp_path = tmp_file.name

        st.video(tmp_path)

        with st.spinner(f"Sampling {num_frames} frames and analyzing..."):
            try:
                video_result = predict_video(tmp_path, model=model, num_frames=num_frames)
            except ValueError as e:
                st.error(str(e))
                video_result = None

        if video_result is not None:
            st.markdown("---")
            label_color = "🟢" if video_result["label"] == "Real" else "🔴"
            st.subheader(f"{label_color} Overall Prediction: **{video_result['label']}**")
            st.metric("Average Confidence", f"{video_result['confidence']}%")
            st.caption(
                f"Analyzed {video_result['num_frames_analyzed']} frames — "
                f"{video_result['fake_frame_votes']} voted Fake, "
                f"{video_result['real_frame_votes']} voted Real."
            )

            st.markdown("**Most representative frame:**")
            rep = video_result["representative_frame"]
            c1, c2, c3 = st.columns(3)
            c1.image(rep["face_crop"], caption="Detected Face", use_container_width=True)
            c2.image(rep["gradcam_overlay"], caption="Grad-CAM", use_container_width=True)
            c3.image(rep["attention_overlay"], caption="ViT Attention", use_container_width=True)

        os.unlink(tmp_path)

# --------------------------------------------------------------------------
# About tab
# --------------------------------------------------------------------------
with tab_about:
    st.markdown(
        f"""
### About this system

This application implements the deepfake detection pipeline described in
the project abstract:

- **CNN branch** (`{config.CNN_BACKBONE}`) extracts local spatial features
  and subtle visual artifacts (blending seams, texture inconsistencies).
- **ViT branch** (`{config.VIT_BACKBONE}`) models global contextual
  relationships across the face and background.
- Features are fused with a **gated fusion layer** and classified by an
  MLP head.
- **Grad-CAM** explains the CNN branch's decision; **attention rollout**
  explains the ViT branch's decision.

**Training datasets supported:** {", ".join(config.SUPPORTED_DATASETS)}

**Evaluation metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC,
Confusion Matrix (see `src/evaluate.py`).

**Disclaimer:** No detector is perfect. Treat predictions as a decision
aid, not definitive proof, especially as generation techniques evolve.
"""
    )
