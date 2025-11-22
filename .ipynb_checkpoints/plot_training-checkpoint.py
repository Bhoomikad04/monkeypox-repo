import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Function to plot the training history
def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 6))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Save the plot
    plt.savefig(f'training_plots/{model_name}_training_plot.png')

# Load the trained models
cnn_model = load_model('models/cnn_model.h5')
resnet_model = load_model('models/resnet_model.h5')
vgg_model = load_model('models/vgg16_model.h5')

# Plot the training history for each model (call this after training each model)
# Example for CNN model:
plot_training_history(cnn_model.history, 'cnn')
