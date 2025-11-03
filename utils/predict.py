# Streamlit アプリ内で「学習済みモデルを読み込んで予測する」ための関数

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

@st.cache_resource
def load_trash_model(model_path, train_dir):
    model = load_model(model_path)
    dataset = image_dataset_from_directory(train_dir, image_size=(224, 224), shuffle=False)
    class_names = dataset.class_names
    return model, class_names

def predict_image(uploaded_image, model, class_names):
    img = Image.open(uploaded_image).convert("RGB").resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    pred = model.predict(x)
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred)
    return predicted_class, confidence, pred[0]
