# app.py
import streamlit as st
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Monkeypox Classifier", layout="centered")

# Paths
MODELS_DIR = "models"
PLOTS_DIR = "training_plots"

# Load models (best saved ones)
@st.cache_resource
def load_models():
    models = {}
    # try both names: the checkpoint best and the final saved name
    cand = {
        "CNN": ["cnn_model.h5", "cnn_best.h5", "cnn_model.h5"],
        "ResNet50": ["resnet_model.h5", "resnet50_best.h5"],
        "VGG16": ["vgg16_model.h5", "vgg16_best.h5"]
    }
    for k, v in cand.items():
        model_path = None
        for name in v:
            p = os.path.join(MODELS_DIR, name)
            if os.path.exists(p):
                model_path = p
                break
        if model_path:
            models[k] = load_model(model_path)
        else:
            models[k] = None
    return models

models = load_models()

CLASS_NAMES = ['Chickenpox', 'Measles', 'Monkeypox', 'Normal']

st.title("🩺 Monkeypox & Similar Skin Disease Classifier")
st.markdown("Upload a skin lesion image and choose a model to predict the label.")

st.sidebar.header("Options")
model_choice = st.sidebar.selectbox("Select model", ["CNN", "ResNet50", "VGG16"])
show_plots = st.sidebar.checkbox("Show training plots", value=False)

uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    desired_size = (128,128)  # match training code
    img_resized = img.resize(desired_size)
    img_arr = np.array(img_resized)/255.0
    input_tensor = img_arr.reshape(1, desired_size[0], desired_size[1], 3)

    model = models.get(model_choice)
    if model is None:
        st.warning(f"{model_choice} model not found in /models/. Train and save the model first.")
    else:
        preds = model.predict(input_tensor)[0]
        top_idx = int(np.argmax(preds))
        label = CLASS_NAMES[top_idx]
        conf = preds[top_idx]*100

        st.success(f"Prediction: **{label}** ({conf:.2f}% confidence)")

        # Probability bar chart
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, preds, color="skyblue")
        ax.set_ylim([0,1])
        ax.set_ylabel("Probability")
        st.pyplot(fig)

if show_plots:
    st.subheader("Training plots")
    for fname in sorted(os.listdir(PLOTS_DIR)) if os.path.exists(PLOTS_DIR) else []:
        if fname.endswith(".png"):
            st.image(os.path.join(PLOTS_DIR, fname), use_column_width=True)
