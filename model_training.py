# model_training.py
# ------------------
# Trains a CNN for facial emotion recognition using images in /train and /test folders.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
)
from tensorflow.keras.optimizers import Adam

# Ensure TensorFlow-CPU backend
print("TensorFlow version:", tf.__version__)

# Paths to your folders
train_dir = "train"
test_dir = "test"

# Image dimensions
img_size = (48, 48)
batch_size = 64

# Data generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)

# Model architecture
model = Sequential([
    Conv2D(64, (3, 3), activation="relu", input_shape=(48, 48, 1)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation="softmax"),
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model
epochs = 50
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
)

# Save trained model
model.save("face_emotionModel.h5")
print("Model saved as face_emotionModel.h5")
