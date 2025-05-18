"""Este módulo contiene funciones para cargar los modelos preentrenados."""
# Librerías externas
import streamlit as st
import torch
# Librerías locales
from src.models import CNN_3C, CNN_4C

# Cargar el modelo 
@st.cache_resource #Esto permite que solo se vuelva a ejecutar esta función si el parámetro 'model_choice' cambia, de forma que es más eficiente
def load_model(model_choice):
    # """Carga los pesos del modelo seleccionado"""
    if model_choice == "CNN_3C":
        model = CNN_3C(64, 128, 256, 64)
        weights_path = "model/model3C.pth"
    else:
        model = CNN_4C(64, 128, 256, 512, 64)
        weights_path = "model/model4C.pth"
    
    try:

        model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))  # Cambiar a otro dispositivo si es necesario
        model.eval()

        return model
    
    except Exception as e:

        st.error(f"Error al cargar el modelo desde {weights_path}: {e}")

        return None  # Devolver None en caso de error
    
# Obtención de resultados del modelo
def predict_image(model, input_tensor):
    # """Devuelve la probabilidad arrojada por el modelo y las salidas de este"""
    try: 
        #Carga en el modelo
        output = model(input_tensor)
        # Traducimos la salida a probabilidad mediante sigmoid (clasificación binaria)
        probability = torch.sigmoid(output).item()
    except Exception as e:
        st.error("No se pudo abrir la imagen")
        probability = None
    
    return probability, output


