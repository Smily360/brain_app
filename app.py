# === app.py ===
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os
import gdown

# =========================
# Config
# =========================
MODEL_PATH = "model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1x0HRnUKBpxHs9DPXYcZRbE0AcKIoZO5C"  # Direct download link
PASSWORD = "Doctor@2025"

# =========================
# Password protection
# =========================
def check_password():
    """Simple password gate"""
    def password_entered():
        if st.session_state["password"] == PASSWORD:
            st.session_state["password_ok"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_ok"] = False

    if "password_ok" not in st.session_state:
        st.text_input("Enter password:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_ok"]:
        st.text_input("Enter password:", type="password", on_change=password_entered, key="password")
        st.error("❌ Wrong password")
        return False
    else:
        return True

# =========================
# Download model function
# =========================
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model from Google Drive...")
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            st.stop()
        if not os.path.exists(MODEL_PATH):
            st.error("Model file not found after download. Check MODEL_URL or permissions.")
            st.stop()

# =========================
# Load model (cached)
# =========================
@st.cache_resource
def load_trained_model():
    download_model()
    return load_model(MODEL_PATH, compile=False)

# =========================
# Class labels
# =========================
class_labels = ["No Tumor", "Tumor"]

# =========================
# Preprocess uploaded image
# =========================
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))        # Resize to match model input
    image = img_to_array(image)             # Convert to numpy array
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    image = image.astype("float32") / 255.0 # Normalize
    return image

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

if check_password():
    st.title("🧠 Brain Tumor Detection App")
    st.write("Upload an MRI scan and the model will classify whether a tumor is present or not.")

    # Load model
    model = load_trained_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

        processed_image = preprocess_image(image)

        # Predict
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        st.subheader("🔍 Prediction Result")
        st.write(f"**Class:** {class_labels[predicted_class]}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")

        if predicted_class == 0:
            st.success("✅ No Tumor Detected")
        else:
            st.error("⚠️ Tumor Detected")

    st.markdown("---")
    st.caption("Model trained with VGG16 + custom layers on Brain MRI dataset.")
