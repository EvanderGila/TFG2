"""Este módulo contiene funciones para abrir y preprocesar las imágenes."""
from PIL import Image
from torchvision import transforms
import streamlit as st

# Preprocesado de la imagen (El mismo que en el entrenamiento del modelo)
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4718, 0.4628, 0.4176], std=[0.2361, 0.2360, 0.2636])
])

# Abrir la imagen
def open_image(uploaded_image):
    # """Abrir la imagen introducida"""
    try:
        # Abrir imagen y formato RGB
        image = Image.open(uploaded_image).convert("RGB")
        
    except Exception as e:
        st.error("No se pudo abrir la imagen introducida")
        image = None

    return image

# Preprocesado de la imagen
def preprocess_image(image):
    try:
        # Preprocesar la imagen y dejarla en formato tensor
        input_tensor = preprocess(image).unsqueeze(0)  # Como el tensor tiene forma [C, H, W] y Pytorch espera [Batch_size, C, H, W] se añade una dimensión: [1, C, H, W]
    except Exception as e:
        st.error("No se pudo preprocesar la imagen introducida")
        input_tensor = None

    return input_tensor

