"""Este archivo contiene la aplicación principal en streamlit"""
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from src.models import CNN_3C, CNN_4C

# Configurar que siempre se expanda por defecto para evitar confusiones
st.set_page_config(layout="wide", initial_sidebar_state="expanded")


# Título principal
st.markdown("<h1 style='text-align: center;'>Detección de imágenes sintéticas</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Detección de imágenes generadas por Stable Diffusion</h3>", unsafe_allow_html=True)
#st.title("Detección de imágenes sintéticas")
#st.write("### Detección de imágenes generadas por Stable Diffusion")


# Selección del tipo de modelo a usar mediante una sidebar
with st.sidebar:
    st.sidebar.title("Opciones del modelo")
    model_choice = st.sidebar.selectbox("Selecciona una arquitectura", ["CNN_3C", "CNN_4C"])
    st.write(f"Modelo seleccionado: **{model_choice}**")


st.divider()

#Cargar el modelo (definición de función)
@st.cache_resource #Esta función permite que solo se vuelva a ejecutar esta función si el parámetro 'model_choice' cambia, de forma que es más eficiente
def load_model(model_choice):
    if model_choice == "CNN_3C":
        model = CNN_3C(64, 128, 256, 64)
        weights_path = "model/model3C.pth"
    else:
        model = CNN_4C(64, 128, 256, 512, 64)
        weights_path = "model/model4C.pth"
    
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu"))) #Cambiar a otro dispositivo si es necesario
    model.eval()

    return model

#Carga de modelo
model = load_model(model_choice)

# Preprocesamiento de la imagen (definición de función)
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),  # Tamaño del modelo
    transforms.ToTensor(),       
    transforms.Normalize(mean=[0.4718, 0.4628, 0.4176], std=[0.2361, 0.2360, 0.2636])  # Media y desviación típica del dataset de entrenamiento
])

#Creamos columnas para mayor organización visual
col1, col2 = st.columns([0.60, 0.40]) # Crea dos columnas

with col1:

    # Carga de la imagen
    uploaded_image = st.file_uploader("Sube una imagen para clasificar", type=["jpg", "jpeg", "png"])

    # Si la imagen está cargada
    if uploaded_image is not None:
    
        # Elegir la imagen y mostrarle
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Imagen subida", use_container_width=True)
        # Preprocesar la imagen
        input_tensor = preprocess(image).unsqueeze(0)  # Como el tensor tiene forma [C, H, W] y Pytorch espera [Batch_size, C, H, W] se añade una dimensión: [1, C, H, W]

        # Evaluación de la imagen
        with torch.no_grad():
            
            # Pasamos la imagen al modelo
            output = model(input_tensor)
            # Traducimos la salida a probabilidad mediante sigmoid (clasificación binaria)
            probability = torch.sigmoid(output).item()

            if probability >= 0.5:
                prediction = "Esta imagen es **real**"
                confidence = probability * 100 # Usamos la propia probabilidad del modelo (1 = Real)
            else:
                prediction = "Esta imagen está **generada sintéticamente (FAKE)**"
                confidence = (1 - probability) * 100 # Invertimos la probabilidad del modelo (0 = Fake)

with col2:
    st.markdown("<br><br><br><br><br><br><br>", unsafe_allow_html=True)
    if uploaded_image is not None:
        st.markdown("### Resultado:")
        if probability >= 0.5:
            st.success(f"✅ {prediction} con una confianza del `{confidence:.4f}%`")
        else:
            st.error(f"⚠️ {prediction} con una confianza del `{confidence:.4f}%`")


# Footer
st.markdown("---")
st.markdown("TFG2")


