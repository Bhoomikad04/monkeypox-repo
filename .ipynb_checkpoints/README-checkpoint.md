# Monkeypox Disease Classifier

This project uses deep learning models to classify images of skin lesions as Monkeypox, Chickenpox, Measles, or Normal.

## Files:
- **/models/**: Contains trained models (CNN, ResNet50, VGG16).
- **/training_plots/**: Contains training accuracy and loss plots.
- **app.py**: The main Streamlit app for web-based image classification.
- **requirements.txt**: List of project dependencies.

## How to Run:
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the app: `streamlit run app.py`.

## Models:
- **CNN Model**: Custom CNN model trained using Keras.
- **ResNet50**: Pre-trained ResNet50 with fine-tuning.
- **VGG16**: Pre-trained VGG16 with fine-tuning.

## Acknowledgments:
- Data from Monkeypox Skin Image Dataset.
