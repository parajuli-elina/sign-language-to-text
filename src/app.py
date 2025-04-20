import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

# Load model
model_path = os.path.join("..", "models", "trained_model.h5")
model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = sorted(os.listdir(os.path.join("..", "data", "asl_alphabet_train")))

# Streamlit app
st.title("Sign Language to Text Detection App 📸")

# Initialize session state variables
if 'run' not in st.session_state:
    st.session_state.run = False
if 'prediction_text' not in st.session_state:
    st.session_state.prediction_text = ""
if 'prediction_counter' not in st.session_state:
    st.session_state.prediction_counter = 0

start = st.button('Start Webcam')
stop = st.button('Stop Webcam')

if start:
    st.session_state.run = True
    st.session_state.prediction_text = ""
    st.session_state.prediction_counter = 0

if stop:
    st.session_state.run = False

FRAME_WINDOW = st.image([])
prediction_placeholder = st.empty()

cap = None
if st.session_state.run:
    cap = cv2.VideoCapture(0)

while st.session_state.run:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture image from webcam.")
        break

    # Preprocess frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (64, 64))  # match training size
    img_array = np.expand_dims(img_resized, axis=0) / 255.0

    # Predict
    preds = model.predict(img_array, verbose=0)
    pred_label = class_labels[np.argmax(preds)]

    # Update prediction text
    st.session_state.prediction_text += pred_label
    st.session_state.prediction_counter += 1

    # If 10 predictions, reset
    if st.session_state.prediction_counter >= 10:
        st.session_state.prediction_text = ""
        st.session_state.prediction_counter = 0

    # Display video
    FRAME_WINDOW.image(img)

    # Display prediction
    prediction_placeholder.subheader(f"Prediction: {st.session_state.prediction_text}")

    # Stop if button pressed
    if not st.session_state.run:
        break

if cap:
    cap.release()
cv2.destroyAllWindows()
