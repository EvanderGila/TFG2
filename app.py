"""Este archivo contiene la aplicación principal en streamlit"""

# Librerías externas
import streamlit as st

# Librerías locales
from src import gradcam_utils as gcu
from src import model_loading as ml
from src import visualization as vis
from src import preprocess as pr
from src import explanation as expl


# Barra lateral expandida por defecto
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# === TÍTULO PRINCIPAL ===
st.markdown("<h1 style='text-align: center;'>Detección de imágenes sintéticas</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Detección de imágenes generadas por Stable Diffusion</h3>", unsafe_allow_html=True)

# === BARRA LATERAL ===
# Creación barra lateral y selección de modelo
with st.sidebar:
    st.sidebar.title("Opciones del modelo")
    # Selección del modelo
    model_choice = st.sidebar.selectbox("Selecciona una arquitectura", ["CNN_3C", "CNN_4C"])
    st.write(f"Modelo seleccionado: **{model_choice}**")
    # Mostar detalles del modelo
    with st.expander("Detalles del modelo seleccionado"):
        vis.show_model_details(model_choice)
        
    # División dentro de la barra lateral
    st.divider()

st.divider()

# === CARGA DEL MODELO ===
model = ml.load_model(model_choice)
# Detener la ejecución si el modelo no se carga
if model is None:
    st.stop() 

# === INICIALIZACIÓN Grad-CAM torchcam ===
cam_torchcam = gcu.initialize_gradcam(model, model_choice)

# === CARGA DE LA IMAGEN ===
uploaded_image = st.file_uploader("Sube una imagen para clasificar", type=["jpg", "jpeg", "png"])

# === PREPROCESADO Y EVALUACIÓN ===
if uploaded_image is not None:
    # Abrir imagen mediante el módulo preprocess
    image = pr.open_image(uploaded_image)

    if image is not None:
        # Preprocesar la imagen y convertirla a tensor mediante el módulo preprocess    
        input_tensor = pr.preprocess_image(image)

        if input_tensor is not None:
            # Obtención de probabilidades y salidas mediante el módulo model_loading
            probability, output = ml.predict_image(model, input_tensor)

            # Conseguir los valores de 'prediction' y 'confidence' del módulo de visualización
            prediction, confidence = vis.display_prediction(probability)
        
            # Mostar resultado inicial del modelo (prediction y confidence)
            st.markdown("### Resultado:")
            if probability >= 0.5:
                st.success(f"#### ✅ {prediction} con una confianza del **{confidence:.4f}%**")
            else:
                st.error(f"#### ⚠️ {prediction} con una confianza del **{confidence:.4f}%**")

#Creamos columnas para mostrar tres imágenes
col1, col2, col3 = st.columns([1, 1, 1]) # Crea tres columnas con proporciones iguales

# === MAPAS DE CALOR ===
# Si hay imagen muestra las columnas y su contenido
if uploaded_image is not None:

    # Columna 1: Muestra la imagen original 
    with col1:
        # Título
        st.markdown("<h4 style='text-align: center;'>Imagen original:</h4>", unsafe_allow_html=True)
        #Mostar la imagen
        st.image(image, caption="Imagen subida", use_container_width=True)
    
    # Columa 2: Muestra el mapa de calor Grad-CAM
    with col2:
        # Título
        st.markdown("<h4 style='text-align: center;'>Mapa Grad-CAM:</h4>", unsafe_allow_html=True)

        # Cambiar 'alpha' del mapa de calor mediante un slider
        alpha = vis.alpha_gradcam()
        # Generacion del mapa de mediante el módulo explanation
        heat_map = expl.generate_gradcam_heatmap(model, cam_torchcam, image, output, alpha)

        #Mostrar el mapa de calor Grad-CAM
        if heat_map is not None:
            # Mostar map de calor
            st.image(heat_map, caption="Mapa Grad-CAM: regiones sensibles al modelo", use_container_width=True)

            # Mostrar botón de descarga
            formato_gcam = st.selectbox("Formato de descarga Grad-CAM", ["PNG", "SVG"], key="formato_gcam")
            
            # Exportar (Módulo de visualización)
            vis.export_imagen_pil(heat_map, "Mapa Grad-CAM", formato_gcam)
        


    # Columa 3: Muestra el mapa de Saliencia
    with col3:
        # Título
        st.markdown("<h4 style='text-align: center;'>Mapa de Saliencia:</h4>", unsafe_allow_html=True)
        
        # Generar el mapa de saliencia mediante el módulo explanation
        saliency_img_resized = expl.generate_saliency_map(model, input_tensor)

        if saliency_img_resized is not None:
            # Mostrar mapa de saliencia
            st.image(saliency_img_resized, caption="Mapa de saliencia: regiones sensibles al modelo", use_container_width=True)
            # Mostrar botón de descarga
            formato_sal = st.selectbox("Formato de descarga Saliencia", ["PNG", "SVG"], key="formato_sal")
            # Exportar (Módulo de visualización)
            vis.export_imagen_pil(saliency_img_resized, "Mapa de Saliencia", formato_sal)

# Divisor de las imágenes y el quesito de probabilidades
st.divider()

# === ESTADÍSTICAS ===
# Si se ha subido la imagen, mostrar probabilidades de esta
if uploaded_image is not None:
    # Crear las columnas
    col4, col5, col6 = st.columns([0.25, 0.5, 0.25]) # Crea 3 columnas
    
    # Columna 1: Espacio en blanco (Estética)
    with col4:
        # Espacio en blanco
        st.markdown("")

    # Columna 2: Gráfico tipo quesito de probabilidades/ probabilidades a secas
    with col5:
        # Opción para mostrar/ocultar el gráfico
        mostrar_grafico = st.sidebar.checkbox("Mostrar gráfico de distribución de probabilidad", value=False)

        # Mostrando el gráfico:
        if mostrar_grafico:
            fig_graph = vis.display_probability_chart(probability)

        # No mostar gráfico:
        else:
            # Mostar probabilidad extendida (Módulo visualización)
            vis.display_probability_text_extended(probability)

    # Columna 6: Mostar estadísticas y descarga del gráfico
    with col6:

        # Si se selecciona el checkbox:
        if mostrar_grafico:
            # Mostar probabilidad (Módulo visualización)
            vis.display_probability_text(probability)

            # Exportar gráfico circular en formato PNG y SVG
            vis.export_graph(fig_graph)
            
# === FOOTER ===
st.markdown("---")
st.markdown("TFG2")


