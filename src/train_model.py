# src/train_model.py

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'asl_alphabet_train')
TEST_DIR = os.path.join(BASE_DIR, 'data', 'asl_alphabet_test')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'trained_model.h5')

# Create models directory if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Image parameters
IMG_WIDTH = 64
IMG_HEIGHT = 64
BATCH_SIZE = 32
EPOCHS = 10

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% for validation
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Number of classes
num_classes = len(train_generator.class_indices)
class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}  # Reverse mapping

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# Save the model
model.save(MODEL_PATH)
print(f"✅ Model trained and saved at {MODEL_PATH}")

# ---------------------------------------------------------------
# Load and Evaluate on Test Dataset (asl-alphabet-test)
# ---------------------------------------------------------------

print("🔎 Evaluating on test images...")

correct = 0
total = 0

for filename in os.listdir(TEST_DIR):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        label = filename.split('_')[0]  # filenames like 'A_test.jpg' -> 'A'
        img_path = os.path.join(TEST_DIR, filename)

        # Load image
        img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # (1, 64, 64, 3)

        # Prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels[predicted_class]

        if predicted_label == label:
            correct += 1
        total += 1

# Final Accuracy
if total > 0:
    accuracy = (correct / total) * 100
    print(f"🎯 Test Accuracy on {total} samples: {accuracy:.2f}%")
else:
    print("⚠️ No test images found.")

