import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model

# Load models
cnn_model = load_model('models/cnn_model.h5')
resnet_model = load_model('models/resnet_model.h5')
vgg_model = load_model('models/vgg16_model.h5')

# Class names
class_names = ['Chickenpox', 'Measles', 'Monkeypox', 'Normal']

# Title and Instructions
st.title("Monkeypox Disease Classifier")
st.write("Upload an image to predict whether it's Monkeypox, Chickenpox, Measles, or Normal.")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Image Preprocessing
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    # Model selection
    model_choice = st.selectbox("Select Model", ("CNN", "ResNet50", "VGG16"))

    if model_choice == "CNN":
        prediction = cnn_model.predict(img_array)
    elif model_choice == "ResNet50":
        prediction = resnet_model.predict(img_array)
    else:
        prediction = vgg_model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: **{predicted_class}** with {confidence:.2f}% confidence.")

    # Display probability chart
    st.subheader("Class Probabilities")
    fig, ax = plt.subplots()
    ax.bar(class_names, prediction[0], color='skyblue')
    ax.set_ylabel("Probability")
    st.pyplot(fig)
