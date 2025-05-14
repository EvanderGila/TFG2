"""Este archivo contiene la aplicación principal en streamlit"""
import streamlit as st
import torch
from torchvision import transforms
from torchvision.transforms import Resize
from PIL import Image
from src.models import CNN_3C, CNN_4C
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image


# Configurar que siempre se expanda por defecto para evitar confusiones
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Título principal
st.markdown("<h1 style='text-align: center;'>Detección de imágenes sintéticas</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Detección de imágenes generadas por Stable Diffusion</h3>", unsafe_allow_html=True)


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
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))  # Cambiar a otro dispositivo si es necesario
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None  # Devolver None en caso de error

# Carga de modelo
model = load_model(model_choice)
if model is None:
    st.stop() # Detiene la ejecución si el modelo no se carga

# Preprocesamiento de la imagen (definición de función)
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),  # Tamaño del modelo
    transforms.ToTensor(),       
    transforms.Normalize(mean=[0.4718, 0.4628, 0.4176], std=[0.2361, 0.2360, 0.2636])  # Media y desviación típica del dataset de entrenamiento
])

# Inicializar Grad-CAM (definición de función)
def initialize_gradcam(model, model_choice):
    if model_choice == "CNN_3C":
        target_layer = "conv3"  # Última capa convolucional del modelo 3C
    else:
        target_layer = "conv4"  # Última capa convolucional del modelo 4C
    return GradCAM(model, target_layer=target_layer)

# Limpiar los hooks (definición de función)
def clear_gradcam_hooks(model):
    for module in model.modules():
        if hasattr(module, 'registered_hooks'):
            for hook in module.registered_hooks:
                hook.remove()

# Inicializar Grad-CAM torchcam
cam_torchcam = initialize_gradcam(model, model_choice)
heat_map = None

# Carga de la imagen
uploaded_image = st.file_uploader("Sube una imagen para clasificar", type=["jpg", "jpeg", "png"])


if uploaded_image is not None:
    # Elegir la imagen
    image = Image.open(uploaded_image).convert("RGB")
    # Preprocesar la imagen
    input_tensor = preprocess(image).unsqueeze(0)  # Como el tensor tiene forma [C, H, W] y Pytorch espera [Batch_size, C, H, W] se añade una dimensión: [1, C, H, W]

    #Carga en el modelo
    output = model(input_tensor)
    # Traducimos la salida a probabilidad mediante sigmoid (clasificación binaria)
    probability = torch.sigmoid(output).item()
    # Clasificación de la imagen y confianza
    if probability >= 0.5:
        prediction = "Esta imagen es **real**"
        confidence = probability * 100 # Usamos la propia probabilidad del modelo (1 = Real)
    else:
        prediction = "Esta imagen está **generada sintéticamente (FAKE)**"
        confidence = (1 - probability) * 100 # Invertimos la probabilidad del modelo (0 = Fake)
    
    st.markdown("### Resultado:")
    if probability >= 0.5:
        st.success(f"#### ✅ {prediction} con una confianza del **{confidence:.4f}%**")
    else:
        st.error(f"#### ⚠️ {prediction} con una confianza del **{confidence:.4f}%**")

#Creamos columnas para mostrar los tres resultados
col1, col2, col3 = st.columns([1, 1, 1]) # Crea tres columnas

# Si la imagen está cargada
if uploaded_image is not None:
    # Columna 1
    with col1:
        st.markdown("<h4 style='text-align: center;'>Imagen original:</h4>", unsafe_allow_html=True)
        #Mostar la imagen
        st.image(image, caption="Imagen subida", use_container_width=True)
    
    # Columa 2
    with col2:
        st.markdown("<h4 style='text-align: center;'>Mapa Grad-CAM:</h4>", unsafe_allow_html=True)
        # Evaluación de la imagen

        # Activación del extractor
        activation_map = cam_torchcam(0, output)
        #activation_map = cam_extractor(predicted_class, output)
        
        # Limpiar hooks para permitir cambio de imágenes en la misma sesión
        clear_gradcam_hooks(model)
        
        #Convertir la imagen original y la máscara PIL y superponer
        resized_img = transforms.Resize((64, 64))(image)
        heat_map = overlay_mask(resized_img, to_pil_image(activation_map[0].detach(), mode='F'), alpha=0.5)

        #Mostrar el mapa de calor Grad-CAM
        if heat_map is not None:
            st.image(heat_map, caption="Grad-CAM: regiones sensibles al modelo", use_container_width=True)

    with col3:
        st.markdown("<h4 style='text-align: center;'>Mapa de Saliencia:</h4>", unsafe_allow_html=True)

        # Copiamos el tensor de entrada de la imagen para no modificar el original
        image_tensor = input_tensor.clone().detach()
        #Calculamos  los gradientes del tensor en retropropagación
        image_tensor.requires_grad_()

        # Calculamos las salidas del modelo
        output_saliency = model(image_tensor)
        #Obtenemos el valor de salida [batch_size, num_classes], siendo el tamaño del lote de 1 y la 'predicted _class' de 0 o 1
        score = output_saliency[0, 0]
        # Realiza la retrorpopagación calculando los gradientes y almacenándolo en el atributo '.grad' del tensor 'image_tensor'
        score.backward()

        # Saliency = max gradiente en cada canal (por pixel)
        saliency = image_tensor.grad.data.abs().squeeze().max(dim=0)[0]

        # Normalizar el tensor de saliencia al rango [0, 255]
        saliency_normalized = ((saliency - saliency.min()) / (saliency.max() - saliency.min()))

        # Convertimos a imagen PIL para mostrar y redimensionamos
        saliency_img = to_pil_image(saliency_normalized, mode='L').convert('RGB')
        saliency_img_resized = transforms.Resize((64, 64))(saliency_img)

        #Mostramos el mapa de saliencia
        st.image(saliency_img_resized, caption="Mapa de saliencia: regiones sensibles al modelo", use_container_width=True)

st.divider()

st.markdown("### **Añadir registro de imágenes o algo parecido**")



# Footer
st.markdown("---")
st.markdown("TFG2")


