import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('leaf_disease_model.keras')

# Class names (ensure these match your model's output classes)
class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
               'Gall Midge', 'Healthy', 'Powdery Mildew', 'Scooty Mould']  # Replace with your actual class names


def predict_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    return predicted_class, confidence


# Streamlit app
st.title("Mango Leaf Disease Detection")
st.write("Upload an image of a mango leaf to detect disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    st.write("Classifying...")
    try:
        predicted_class, confidence = predict_image(img)

        if predicted_class < len(class_names):
            st.write(f"Prediction: {class_names[predicted_class]} ({confidence * 100:.2f}%)")
        else:
            st.write(f"Prediction index out of range: {predicted_class}")

        st.write(f"Raw Predictions: {predictions}")
    except Exception as e:
        st.write(f"Error: {e}")
