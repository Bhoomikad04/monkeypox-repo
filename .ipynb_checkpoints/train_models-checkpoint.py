import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Setup paths
dataset_dir = 'dataset'

# Create ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(os.path.join(dataset_dir, 'train'), target_size=(224, 224), batch_size=32, class_mode='categorical')
validation_generator = val_test_datagen.flow_from_directory(os.path.join(dataset_dir, 'val'), target_size=(224, 224), batch_size=32, class_mode='categorical')

# CNN Model (Custom)
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(train_generator, validation_data=validation_generator, epochs=5)
cnn_model.save('models/cnn_model.h5')

# ResNet50 Model (Transfer Learning)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output = Dense(4, activation='softmax')(x)

resnet_model = Model(inputs=base_model.input, outputs=output)
resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
resnet_model.fit(train_generator, validation_data=validation_generator, epochs=5)
resnet_model.save('models/resnet_model.h5')

# VGG16 Model (Transfer Learning)
base_model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model_vgg.output)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output_vgg = Dense(4, activation='softmax')(x)

vgg_model = Model(inputs=base_model_vgg.input, outputs=output_vgg)
vgg_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
vgg_model.fit(train_generator, validation_data=validation_generator, epochs=5)
vgg_model.save('models/vgg16_model.h5')
