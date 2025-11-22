# train_models.py
import os
import pickle
from datetime import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, GlobalAveragePooling2D)
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from plot_training import plot_and_save_history

# ========== CONFIG ==========
DATASET_DIR = "dataset"       # parent folder containing train/ val/ test/
IMAGE_SIZE = (128, 128)      # smaller -> faster for dev
BATCH_SIZE = 16
EPOCHS = 25                  # raise later for final runs
NUM_CLASSES = 4              # Chickenpox, Measles, Monkeypox, Normal

MODELS_DIR = "models"
ARTIFACTS_DIR = "artifacts"
PLOTS_DIR = "training_plots"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ========== DATA GENERATORS ==========
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)
val_test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)
validation_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# compute class weights to balance training if dataset is imbalanced
train_labels = train_generator.classes
class_weights_arr = compute_class_weight(class_weight="balanced",
                                         classes=np.unique(train_labels),
                                         y=train_labels)
class_weights = {i: w for i, w in enumerate(class_weights_arr)}
print("Class indices:", train_generator.class_indices)
print("Class weights:", class_weights)

# common callbacks
def get_callbacks(model_name):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    chk = ModelCheckpoint(filepath=os.path.join(MODELS_DIR, f"{model_name}_best.h5"),
                          monitor="val_loss", save_best_only=True)
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3)
    return [chk, es, rlr]

# ========== 1) CNN (custom) ==========
def build_cnn(input_shape=(128,128,3), num_classes=4):
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    return model

print("Building and training CNN...")
cnn = build_cnn(input_shape=(*IMAGE_SIZE,3), num_classes=NUM_CLASSES)
cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy", metrics=["accuracy"])
history_cnn = cnn.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=get_callbacks("cnn"),
    verbose=1
)
cnn.save(os.path.join(MODELS_DIR, "cnn_model.h5"))
with open(os.path.join(ARTIFACTS_DIR, "cnn_history.pkl"), "wb") as f:
    pickle.dump(history_cnn.history, f)
plot_and_save_history(history_cnn.history, "cnn", PLOTS_DIR)

# ========== 2) ResNet50 (transfer learning) ==========
print("Building and training ResNet50 (transfer learning)...")
base_resnet = ResNet50(weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE,3))
# freeze base initially
for layer in base_resnet.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_resnet.output)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)
resnet = Model(inputs=base_resnet.input, outputs=out)

resnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
               loss="categorical_crossentropy", metrics=["accuracy"])
history_resnet = resnet.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=get_callbacks("resnet50"),
    verbose=1
)
resnet.save(os.path.join(MODELS_DIR, "resnet_model.h5"))
with open(os.path.join(ARTIFACTS_DIR, "resnet_history.pkl"), "wb") as f:
    pickle.dump(history_resnet.history, f)
plot_and_save_history(history_resnet.history, "resnet50", PLOTS_DIR)

# Optional: fine-tune top layers (unfreeze last few blocks) — uncomment if desired
# for layer in base_resnet.layers[-30:]:
#     layer.trainable = True
# recompile with lower LR and continue training

# ========== 3) VGG16 (transfer learning) ==========
print("Building and training VGG16 (transfer learning)...")
base_vgg = VGG16(weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE,3))
for layer in base_vgg.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_vgg.output)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)
vgg = Model(inputs=base_vgg.input, outputs=out)

vgg.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy", metrics=["accuracy"])
history_vgg = vgg.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=get_callbacks("vgg16"),
    verbose=1
)
vgg.save(os.path.join(MODELS_DIR, "vgg16_model.h5"))
with open(os.path.join(ARTIFACTS_DIR, "vgg_history.pkl"), "wb") as f:
    pickle.dump(history_vgg.history, f)
plot_and_save_history(history_vgg.history, "vgg16", PLOTS_DIR)

print("Training complete. Models saved to:", MODELS_DIR)
