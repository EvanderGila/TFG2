"""Este módulo contiene funciones para la generación de mapas de explicación visual (Grad-CAM, Saliencia)"""
# Librerias externas
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchcam.utils import overlay_mask
import streamlit as st

# Librerias locales
from src import gradcam_utils as gcu

# Generación del mapa Grad-CAM
def generate_gradcam_heatmap(model, cam_torchcam, image, output):
    # """Genera el mapa de calor Grad-CAM"""
    try: 
        # Activación del extractor (siendo 0 el índice de la clase objetivo ya que solo hay una y output a salida del modelo)
        activation_map = cam_torchcam(0, output)
        
        # Limpiar hooks 
        gcu.clear_gradcam_hooks(model)
        
        #Convertir la imagen original y la máscara PIL y superponer
        resized_img = transforms.Resize((64, 64))(image)
        # activation_map[0] (Accede al primer y único mapa generado) .detach() (Devuelve una copia del tensor original que no está conectado a la gráfica de cálculo para evitar problemas)
        heat_map = overlay_mask(resized_img, to_pil_image(activation_map[0].detach(), mode='F'), alpha=0.5)
        
    except Exception as e:
        st.error(f"Error al generar el mapa Grad-CAM: {e}")
        heat_map = None

    return heat_map

# Generar el mapa de saliencia
def generate_saliency_map(model, input_tensor):
    # """Genera el mapa de Saliencia"""
    try:
        # Copiamos la imagen en forma de tensor para no modificar el original y lo separamos de  la gráfica de cálculo
        image_tensor = input_tensor.clone().detach()
        # Activamos el seguimiento de los gradientes
        image_tensor.requires_grad_()

        # Calculamos las salidas del modelo
        output_saliency = model(image_tensor)
        # Obtenemos el valor de salida [batch_size, num_classes], siendo el tamaño del lote de 1 (0) y la 'predicted _class' de 0 porque solo hay una neurona (clase)
        score = output_saliency[0, 0]
        # Realiza la retrorpopagación calculando los gradientes (Calculando la derivada de la salida con respecto a cada entrada (píxel)) y almacenándolo en el atributo '.grad' del tensor 'image_tensor'
        score.backward()

        # Cáclulo del mapa de saliencia, grad.data.abs() calcula el valor absoluto de los gradientes, .squeeze() elimina la dimensión extra y max(dim=0)[0] para cada píxel (x,y), toma el canal con mayor gradiente (R,G,B)
        saliency = image_tensor.grad.data.abs().squeeze().max(dim=0)[0]

        # Normaliza todos los valores al rango [0, 1] para poder visualizarlos como imagen
        saliency_normalized = ((saliency - saliency.min()) / (saliency.max() - saliency.min()))

        # Convertimos a imagen PIL para mostrar y redimensionamos
        saliency_img = to_pil_image(saliency_normalized, mode='L').convert('RGB')
        saliency_img_resized = transforms.Resize((64, 64))(saliency_img)
    except Exception as e:
        st.error(f"Error al generar el mapa de Saliencia: {e}")
        saliency_img_resized = None

    return saliency_img_resized
