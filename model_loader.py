import os
import gdown
import streamlit as st

# Google Drive file IDs (extract from shareable links)
MODEL_URLS = {
    "Custom CNN": "https://drive.google.com/uc?id=YOUR_FILE_ID_1",
    "EfficientNet-B0": "https://drive.google.com/uc?id=YOUR_FILE_ID_2", 
    "DenseNet121": "https://drive.google.com/uc?id=YOUR_FILE_ID_3",
    "VGG16": "https://drive.google.com/uc?id=YOUR_FILE_ID_4",
    "ViT-Base-16": "https://drive.google.com/uc?id=YOUR_FILE_ID_5",
    "DeiT-Small-16": "https://drive.google.com/uc?id=YOUR_FILE_ID_6"
}

@st.cache_resource
def download_model(model_name):
    """Download model weights if not already cached"""
    model_path = f"models/{model_name.replace(' ', '_').lower()}.h5"
    
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        
        with st.spinner(f"🔄 Downloading {model_name} model (first time only)..."):
            gdown.download(MODEL_URLS[model_name], model_path, quiet=False)
            
    return model_path

# Usage in your main app:
def load_model(model_config):
    model_path = download_model(model_config['name'])
    # Load your model using tensorflow.keras.models.load_model or torch.load
    return loaded_model
