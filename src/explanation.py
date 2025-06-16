"""Este módulo contiene funciones para la generación de mapas de explicación visual (Grad-CAM, Saliencia, LIME)"""

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
        # Cargamos el modelo dentro de la función generate_lime_explanation para eficiencia de caché
        model = load_model(model_choice)
        # Ponemos el modelo en mode evaluación
        model.eval() 

        # Se crea el explicador LIME para imágenes
        explainer = LimeImageExplainer()

        width, height = image.size
        if width != 32 and  height != 32 :
            img = transforms.Resize((64, 64))(image)
        else:
            img = image

        # Convertir a formato uint8 la imagen si no lo está
        np_image = np.array(img)
        if np_image.dtype != np.uint8:
            np_image = (np_image * 255).astype(np.uint8)

        # Segmentador personalizado,  SLIC (Simple Linear Iterative Clustering) n_segments= número de superpíxeles, compactness= relación entre el color y la proximidad espacial, start_label asigna la etiqueta del primer superpixel
        segmentation_fn = lambda x: slic(x, n_segments=90, compactness=20, start_label=1)
       
        explanation = explainer.explain_instance(
            np_image, # Imagen
            classifier_fn=lambda x: predict_fn(x, model, preprocess_fn), # Usamos predict_fn para sacar las probabilidades de cada clase para las imágenes de lime
            top_labels=1, # Enfocarse en la clase que tiene más probabilidades (En este caso solo hay una)
            num_samples=600, # Número de muestras perturbadas
            segmentation_fn=segmentation_fn # Función de segmentación definida previamente (Crea los super píxeles)
        )

        # Obtener la etiqueta de la clase predicha para la explicación LIME
        predicted_class_idx = int(explanation.top_labels[0])
        # Usar class_names para obtener el nombre de la clase predicha
        predicted_class_name = class_names[predicted_class_idx]

        # inicializamos las variables de imagen final y máscara de esa imagen
        final_image_to_display = None
        final_mask_for_boundaries = None

        # Obtener los superpíxeles relevantes y sus pesos (los 7 más importantes)
        top_features = explanation.local_exp[predicted_class_idx][:7]
        # Crear una máscara booleana de TODOS los superpíxeles que LIME considera relevantes (positivos o negativos)
        all_relevant_segments_mask = np.zeros(explanation.segments.shape, dtype=bool)
        for superpixel_id, _ in top_features:
            all_relevant_segments_mask[explanation.segments == superpixel_id] = True

        # Crear una máscara de enteros para mark_boundaries.
        initial_mask_for_boundaries = np.zeros(explanation.segments.shape, dtype=int)
        for superpixel_id, weight in top_features:
            segment_pixels = (explanation.segments == superpixel_id)
            if weight > 0:
                initial_mask_for_boundaries[segment_pixels] = 1 # Positivo (borde verde)
            else: # weight <= 0
                initial_mask_for_boundaries[segment_pixels] = -1 # Negativo (borde rojo)

        if hide_rest_option:
            # Viusalizar la  imagen oculta  con los superpíxeles relevantes super puestos

            # Creamos el fondo
            colored_background = np.full(np_image.shape, hide_color_option, dtype=np.uint8)
            final_image_to_display = np.copy(colored_background)

            # Identificar superpíxeles que realmente queremos mostrar
            segments_to_show_original = np.zeros(explanation.segments.shape, dtype=bool)
            
            if predicted_class_name == "Fake":
                # Si es fake, solo mostrar los superpíxeles que contribuyen negativamente (rojos) para no mostar los que contribuyen a la clase real (verdes)
                for superpixel_id, weight in top_features:
                    if weight <= 0: # Solo si contribuyen negativamente 
                        segments_to_show_original[explanation.segments == superpixel_id] = True
            else:
                # Si es real, mostrar todos los superpíxeles relevantes que contribuyen positivamente (verdes)
                for superpixel_id, weight in top_features:
                    if weight >= 0: # Solo si contribuyen positivamente 
                        segments_to_show_original[explanation.segments == superpixel_id] = True

            # Mostrar en la imagen original solo los  superpíxeles relevantes para la clase objetivo
            final_image_to_display[segments_to_show_original] = np_image[segments_to_show_original]
            # Rodear solo los superpíxeles relevantes para la calse objetivo
            final_mask_for_boundaries = np.copy(initial_mask_for_boundaries)


        else:
            # Visualizar la imagen original con los superpíxeles superpuestos
            temp_lime_colored, mask_lime_output = explanation.get_image_and_mask(
                label=predicted_class_idx, # Id de la clase predicha
                positive_only=False, # Necesitamos ambos para el procesamiento, positivos y negativos
                hide_rest=False, # Ocultar el resto de la imagen (falso)
                num_features=7, # Número de superpíxeles a mostrar
                min_weight=0.0 # Umbral de superpíxeles
            )

            if predicted_class_name == "Fake":
                # Si es Fake, quitamos el coloreado de los superpíxeles que contribuyen a 'Real' (positivos)
                for superpixel_id, weight in top_features:
                    if weight > 0: # Este superpíxel contribuye a la clase 'Real'
                        segment_pixels = (explanation.segments == superpixel_id)
                        # Reemplazamos los píxeles coloreados de este superpíxel con la imagen original
                        temp_lime_colored[segment_pixels] = np_image[segment_pixels]
                        # También los ponemos a 0 en la máscara para mark_boundaries para que no tengan borde
                        mask_lime_output[segment_pixels] = 0
            else:
                # Si es Real, quitamos el coloreado de los superpíxeles que contribuyen a 'Fake' (negativos)
                for superpixel_id, weight in top_features:
                    if weight < 0: # Este superpíxel contribuye a la clase 'Fake'
                        segment_pixels = (explanation.segments == superpixel_id)
                        # Reemplazamos los píxeles coloreados de este superpíxel con la imagen original
                        temp_lime_colored[segment_pixels] = np_image[segment_pixels]
                        # También los ponemos a 0 en la máscara para mark_boundaries para que no tengan borde
                        mask_lime_output[segment_pixels] = 0


            # Establecemos la imagen final y la máscara de bordes
            final_image_to_display = temp_lime_colored
            final_mask_for_boundaries = mask_lime_output


        # Quitamos los bordes de la clase opuesta
        if predicted_class_name == "Fake":
            # Eliminar los bordes de la clase real
            final_mask_for_boundaries[final_mask_for_boundaries == 1] = 0
        else:
            # Eliminar los bordes de la clase fake
            final_mask_for_boundaries[final_mask_for_boundaries == -1] = 0

        # Normalizar la imagen final a [0, 1] y asegurar el dtype float32 para mark_boundaries
        final_image_to_display_normalized = final_image_to_display.astype(np.float32) / 255.0

        return mark_boundaries(final_image_to_display_normalized, final_mask_for_boundaries), predicted_class_name

    except Exception as e:
        st.error(f"Error al generar la explicación LIME: {e}")
        return None, None
