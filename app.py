import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import io

# Load the pre-trained MobileNetV2 model
@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

model = load_model()

# Function to add custom CSS for styling
def add_custom_css():
    st.markdown(
        """
        <style>
        /* Background image */
        .stApp {
            background-image: url("https://www.example.com/your-background-image.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        /* Header style */
        .header {
            text-align: center;
            color: #ffffff;
            font-size: 40px;
            font-weight: bold;
            margin-bottom: 20px;
            padding: 20px;
            background: rgba(0, 0, 0, 0.7);
            border-radius: 10px;
        }

        /* Custom container style */
        .container {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.3);
            margin-top: 20px;
            margin-bottom: 20px;
        }

        /* Prediction text style */
        .prediction {
            font-size: 20px;
            font-weight: bold;
            color: #444;
        }

        /* Center alignment */
        .center {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
        }

        /* File uploader styling */
        .file-uploader {
            margin: 20px auto;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply custom CSS
add_custom_css()

# Streamlit app layout
st.markdown('<div class="header">Image Classification with MobileNetV2</div>', unsafe_allow_html=True)

st.write(
    "Upload an image and the model will classify it. This demo uses the MobileNetV2 model trained on ImageNet."
)

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type="jpg", key="file_uploader")

if uploaded_file is not None:
    # Load the image
    image_data = uploaded_file.read()
    img = Image.open(io.BytesIO(image_data))

    # Display the image
    st.image(img, caption='Uploaded Image', use_column_width=True, output_format='JPEG')

    st.write("Classifying...")

    # Prepare the image for classification
    img = img.resize((224, 224))  # Resize image to the size MobileNetV2 expects
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image

    # Predict the class
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Display the result
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.write("**Prediction results:**")
    
    for i, (class_id, class_name, score) in enumerate(decoded_predictions):
        st.markdown(f'<p class="prediction">{i + 1}. **{class_name}**: {score:.2f} ({score*100:.1f}%)</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
