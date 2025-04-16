import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import tempfile
import os
from io import BytesIO
from streamlit_cropper import st_cropper
import matplotlib.pyplot as plt
import seaborn as sns
import gdown

st.set_page_config(
    page_title="Face Shape Classifier & Glasses Recommendation",
    page_icon="👓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use relative paths or download from a public URL
FACE_SHAPES = {
    "Oval": "🧺",
    "Round": "⚪",
    "Square": "🟧",
    "Heart": "❤️",
    "Diamond": "💎"
}

recommendations = {
    "Oval": {
        "description": "Oval faces are balanced and versatile. Almost any frame style works well!",
        "best_frames": ["Round", "Aviator", "Wayfarer", "Rectangle"],
        "avoid": "Oversized frames that can overwhelm your balanced proportions",
        "celebrity_examples": ["Beyoncé", "George Clooney", "Ryan Reynolds"]
    },
    "Round": {
        "description": "Round faces benefit from frames that add angles and definition.",
        "best_frames": ["Square", "Rectangle", "Browline", "Cat-eye"],
        "avoid": "Small, round frames that emphasize roundness",
        "celebrity_examples": ["Emma Stone", "Leonardo DiCaprio", "Selena Gomez"]
    },
    "Square": {
        "description": "Square faces have strong jawlines and need frames to soften angles.",
        "best_frames": ["Round", "Oval", "Aviator", "Rimless"],
        "avoid": "Angular, boxy frames that emphasize squareness",
        "celebrity_examples": ["Angelina Jolie", "Brad Pitt", "Keira Knightley"]
    },
    "Heart": {
        "description": "Heart-shaped faces need frames that balance the wider forehead.",
        "best_frames": ["Light-colored", "Rimless", "Bottom-heavy", "Round"],
        "avoid": "Top-heavy or decorative frames that draw attention upward",
        "celebrity_examples": ["Reese Witherspoon", "Ryan Gosling", "Scarlett Johansson"]
    },
    "Diamond": {
        "description": "Diamond/Triangle faces need frames that highlight cheekbones and soften angles.",
        "best_frames": ["Oval", "Cat-eye", "Rimless", "Semi-rimless"],
        "avoid": "Narrow or boxy frames that don't complement cheekbones",
        "celebrity_examples": ["Rihanna", "Johnny Depp", "Zac Efron"]
    }
}

@st.cache_resource
def load_model():
    # Use a public Google Drive link or host the model elsewhere
    model_url = "https://drive.google.com/uc?id=1hqg4T2NJDeFREW9agD7uJR3O-ZuczKsM"
    model_path = "face_classification_final2.h5"
    
    if not os.path.exists(model_path):
        try:
            gdown.download(model_url, model_path, quiet=False)
        except Exception as e:
            st.error(f"Error downloading model: {str(e)}")
            return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

def preprocess_image(image, target_size=(224, 224)):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(target_size)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image, image_array
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def display_results(prediction, original_image, processed_image):
    predicted_index = np.argmax(prediction)
    predicted_class = list(FACE_SHAPES.keys())[predicted_index]
    confidence_score = np.max(prediction) * 100

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction Results")
        st.markdown(f"### Predicted Face Shape: {predicted_class} {FACE_SHAPES[predicted_class]}")
        st.progress(int(confidence_score))
        st.caption(f"Confidence: {confidence_score:.1f}%")

    with col2:
        st.subheader("Recommendations")
        rec = recommendations[predicted_class]
        st.markdown(f"**Description:** {rec['description']}")
        st.markdown(f"**Best frame styles:** {', '.join(rec['best_frames'])}")

    st.image(processed_image, caption="Processed Image", use_column_width=True)

def main():
    st.title("👓 Face Shape Classifier & Glasses Recommendation")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None and model is not None:
        try:
            original_image = Image.open(uploaded_file)
            processed_image, image_array = preprocess_image(original_image)
            
            if processed_image and image_array is not None:
                with st.spinner("Analyzing face shape..."):
                    prediction = model.predict(image_array)
                display_results(prediction, original_image, processed_image)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()