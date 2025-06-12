"""Este módulo contiene funciones para la generación de mapas de explicación visual (Grad-CAM, Saliencia)"""
# Librerias externas
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchcam.utils import overlay_mask
import streamlit as st
from skimage.segmentation import mark_boundaries
import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic
from lime.lime_image import LimeImageExplainer

# Librerias locales
from src import gradcam_utils as gcu
from src.model_loading import load_model
from src.preprocess import preprocess_image as preprocess_fn

# Generación del mapa Grad-CAM
def generate_gradcam_heatmap(model, cam_torchcam, image, output, alpha):
    # """Genera el mapa de calor Grad-CAM"""
    try: 
        # Activación del extractor (siendo 0 el índice de la clase objetivo ya que solo hay una y output a salida del modelo)
        activation_map = cam_torchcam(0, output)
        
        # Limpiar hooks 
        gcu.clear_gradcam_hooks(model)
        
        #Convertir la imagen original y la máscara PIL y superponer
        resized_img = transforms.Resize((64, 64))(image)
        # activation_map[0] (Accede al primer y único mapa generado) .detach() (Devuelve una copia del tensor original que no está conectado a la gráfica de cálculo para evitar problemas)
        heat_map = overlay_mask(resized_img, to_pil_image(activation_map[0].detach(), mode = 'F'), alpha = alpha)
        
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

# Predecir las clases de la imagen para LIME
def predict_fn(images, model, preprocess_fn):
    # """ Función de predicción para LIME adaptada a modelos binarios con sigmoid. Convierte la salida de una sola probabilidad en dos clases [P(falso), P(real)]"""
    model.eval() # Modelo en modo evaluación
    results = [] # Para almacenar las probabilidades de clase predichas para cada imagen
    for img in images:
        # Preprocesar imagen individual, y convertimos los datos de la imagen a uint8 (formato estándar para  las imagenes)
        input_tensor = preprocess_fn(Image.fromarray(img.astype('uint8')))
        
        # Asegurarse de que el tensor tiene forma (1, C, H, W) para PyTorch
        if input_tensor.dim() == 3: # Si el tensor tiene esta forma (C, H, W)
            input_tensor = input_tensor.unsqueeze(0)  # Añadir dimensión batch --> (1, C, H, W)
        # Predecimos la clase de la imagen 
        prob = torch.sigmoid(model(input_tensor)).item()
        # Como las clases son mutuamente excluyentes, clasificamos la probabilidad de Fake y Real en una tupla y lo añadimos a results
        results.append([1 - prob, prob])  # [P(fake), P(real)]

    return np.array(results)

# Genera una explicación LIME
@st.cache_data
def generate_lime_explanation(image, model_choice, class_names=["Fake", "Real"], hide_rest_option=False, hide_color_option=0):
    # """ Función que genera una imagen con la explicación LIME superpuesta"""
    try:
        #Cargamos el modelo dentro de la función para eficiencia de caché
        model = load_model(model_choice)
        # Se crea el explicador LIME para imágenes
        explainer = LimeImageExplainer()

        # Convertir a uint8 si no lo está
        np_image = np.array(image)
        if np_image.dtype != np.uint8:
            np_image = (np_image * 255).astype(np.uint8)

        # Segmentador personalizado,  SLIC (Simple Linear Iterative Clustering) n_segments= número de superpíxeles, compactness= relación entre el color y la proximidad espacial
        segmentation_fn = lambda x: slic(x, n_segments=100, compactness=20)

        explanation = explainer.explain_instance(
            np_image, # Imagen
            classifier_fn=lambda x: predict_fn(x, model, preprocess_fn), # Usamos predict_fn para sacar las probabilidades de cada clase para las imágenes de lime
            top_labels=1, # Enfocarse en la clase que tiene más probabilidades (En este caso solo hay una)
            hide_color=hide_color_option, # Ocultar píxeles con color seleccionado (0 = negro 255= blanco)
            num_samples=2000, # Número de muestras perturbadas
            segmentation_fn=segmentation_fn # Función de segmentación definida previamente (Crea los super píxeles)
        )

        # Obtener la etiqueta de la clase predicha para la explicación LIME
        predicted_class_idx = int(explanation.top_labels[0])
        # Usar class_names para obtener el nombre de la clase predicha
        predicted_class_name = class_names[predicted_class_idx]
        
        temp, mask = explanation.get_image_and_mask(
            label=predicted_class_idx, # Id de la clase predicha
            positive_only=False, # Necesitamos ambos para el procesamiento, positivos y negativos
            hide_rest=hide_rest_option, # Ocultar el resto de la imagen
            num_features=7, # Número de superpíxeles a mostrar
            min_weight=0.0 # Umbral de superpíxeles
        )

        # Si queremos ocultar el fondo
        if hide_rest_option:
            # Creamos un fondo solido obteniendo las dimensiones de 'temp', eligiendo el color de 'hide_color_option' y dejando los valores en el rango 0-255 (uint8)
            colored_background = np.full(temp.shape, hide_color_option, dtype=np.uint8)
            # Identificación de los superpíxeles relevantes
            relevant_segments_indices = explanation.segments[mask != 0] # Cogemos del array explanation.segments los pesos del array mask que sean distintos a 0 (relevantes)
            relevance_mask = np.zeros(explanation.segments.shape, dtype=bool) # Obtenemos las dimensiones de explanation.segments y lo inicializa a 0 (False)
            for seg_idx in relevant_segments_indices:
                relevance_mask[explanation.segments == seg_idx] = True # Pone a True todos los píxeles que sean relevantes (pesos distintos a 0, positivos o negativos)
            # Combinación del fondo con las partes relevantes
            final_img_for_boundaries = np.where(relevance_mask[:,:,None], np_image, colored_background) # Si el pixel es True, se coge  el valor de np_image, si no, se coge el de colored_background
            temp = final_img_for_boundaries

        # Si es una predicción "Fake", manipulamos la 'temp' para quitar el color verde
        if predicted_class_name == "Fake":
            # Identificamos los píxeles de los superpíxeles con peso positivo (verdes)
            positive_segments_mask = np.zeros(explanation.segments.shape, dtype=bool)
            for seg_id, weight in explanation.local_exp[predicted_class_idx]:
                if weight > 0:
                    positive_segments_mask[explanation.segments == seg_id] = True

            # Sobre la imagen 'temp', donde hay un segmento positivo, lo reemplazamos con la imagen original (o con el hide_color si hide_rest_option es True)
            if hide_rest_option: # Si estamos ocultando el resto, revertimos a hide_color_option
                temp = np.where(positive_segments_mask[:, :, None], colored_background, temp)
            else: # Si no estamos ocultando el resto, revertimos a la imagen original
                temp = np.where(positive_segments_mask[:, :, None], np_image, temp)

        # Solo modificamos la máscara de bordes si la clase predicha es "Fake"
        if predicted_class_name == "Fake":
            mask_for_boundaries = np.copy(mask)
            mask_for_boundaries[mask_for_boundaries > 0] = 0 # Quita los bordes verdes
        else:
            mask_for_boundaries = mask

        return mark_boundaries(temp / 255.0, mask_for_boundaries), predicted_class_name

    except Exception as e:
        st.error(f"Error al generar la explicación LIME: {e}")
        return None, None
