"""Este archivo contiene la aplicación principal en streamlit"""
# librerías estándar
import io

# Librerías externas
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

# Librerías locales
from src.models import CNN_3C, CNN_4C



# Barra lateral expandida por defecto
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# === Título principal ===
st.markdown("<h1 style='text-align: center;'>Detección de imágenes sintéticas</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Detección de imágenes generadas por Stable Diffusion</h3>", unsafe_allow_html=True)

# === Barra lateral ===
# Creación barra lateral y selección de modelo
with st.sidebar:
    st.sidebar.title("Opciones del modelo")
    model_choice = st.sidebar.selectbox("Selecciona una arquitectura", ["CNN_3C", "CNN_4C"])
    st.write(f"Modelo seleccionado: **{model_choice}**")
    # División dentro de la barra lateral
    st.divider()

st.divider()

# === Funciones auxiliares ===

# Preprocesamiento de la imagen (definición de función), mismo preprocess que el dataset
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),  # Tamaño del modelo
    transforms.ToTensor(),       
    transforms.Normalize(mean=[0.4718, 0.4628, 0.4176], std=[0.2361, 0.2360, 0.2636])  # Media y desviación típica del dataset de entrenamiento
])
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

# Cambiar formato de imagen y descargar (función)
def exportar_imagen_pil(imagen_pil, nombre_archivo, formato):
    
    # Crear buffer en memoria RAM
    buffer = io.BytesIO()

    # Convertimos imagen PIL a figura matplotlib para exportar como SVG si se requiere
    # Creamos una figura (fig) y un eje (ax)
    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=100)
    # Mostramos la figura dentro del eje
    ax.imshow(imagen_pil)
    # Ocultams elementos del eje
    ax.axis('off')
    # Ajustamos los márgenes sin espacio adicional
    fig.tight_layout(pad=0)

    # Guardamos la imagen en el buffer
    fig.savefig(buffer, format=formato.lower(), bbox_inches='tight', facecolor=fig.get_facecolor())
    # Puntero del buffer al inicio del archivo
    buffer.seek(0)

    # Botón de descarga
    st.download_button(
        label=f"📥 Descargar {nombre_archivo} como {formato}",
        data=buffer,
        file_name=f"{nombre_archivo.lower().replace(' ', '_')}.{formato.lower()}",
        mime="image/png" if formato == "PNG" else "image/svg+xml"
    )
    # Cerramos la figura
    plt.close(fig)
    # Cerramos el buffer
    buffer.close()

# === Carga de modelo ===
model = load_model(model_choice)
# Detener la ejecución si el modelo no se carga
if model is None:
    st.stop() 


# === Inicialización Grad-CAM torchcam ===
cam_torchcam = initialize_gradcam(model, model_choice)
heat_map = None

# === Carga de la imagen ===
uploaded_image = st.file_uploader("Sube una imagen para clasificar", type=["jpg", "jpeg", "png"])

# === Preprocesado y evaluación de la imagen ===
if uploaded_image is not None:
    try:
        # Abrir imagen y formato RGB
        image = Image.open(uploaded_image).convert("RGB")
    except Exception as e:
        st.error("No se pudo abrir la imagen")
        image = None

    if image is not None:
        try:
            # Preprocesar la imagen y dejarla en formato tensor
            input_tensor = preprocess(image).unsqueeze(0)  # Como el tensor tiene forma [C, H, W] y Pytorch espera [Batch_size, C, H, W] se añade una dimensión: [1, C, H, W]
        except Exception as e:
            st.error("No se pudo abrir la imagen")
            input_tensor = None

    if input_tensor is not None:
        try: 
            #Carga en el modelo
            output = model(input_tensor)
            # Traducimos la salida a probabilidad mediante sigmoid (clasificación binaria)
            probability = torch.sigmoid(output).item()
        except Exception as e:
            st.error("No se pudo abrir la imagen")
            probability = None

    if probability is not None:
        # Clasificación de la imagen y confianza según probabilidad
        if probability >= 0.5:
            prediction = "Esta imagen es **real**"
            confidence = probability * 100 # Usamos la propia probabilidad del modelo (1 = Real)
        else:
            prediction = "Esta imagen está **generada sintéticamente (FAKE)**"
            confidence = (1 - probability) * 100 # Invertimos la probabilidad del modelo (0 = Fake)
        
        # Mostar resultado inicial del modelo
        st.markdown("### Resultado:")
        if probability >= 0.5:
            st.success(f"#### ✅ {prediction} con una confianza del **{confidence:.4f}%**")
        else:
            st.error(f"#### ⚠️ {prediction} con una confianza del **{confidence:.4f}%**")

#Creamos columnas para mostrar tres imágenes
col1, col2, col3 = st.columns([1, 1, 1]) # Crea tres columnas con proporciones iguales

# === Mapas de calor de la imagen ===
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
        # Evaluación de la imagen

        #Opción si hay más de dos clases (Softmax)
        #Siendo predicted_class la clase predicha por un modelo softmax:
        #activation_map = cam_torchcam(predicted_class, output)
        #De forma que eliges la clase, diferente a sigmoid que solo tiene un resultado probabilistico
        try: 
            # Activación del extractor (siendo 0 el índice de la clase objetivo ya que solo hay una y output a salida del modelo)
            activation_map = cam_torchcam(0, output)
        
            # Limpiar hooks 
            clear_gradcam_hooks(model)
        
            #Convertir la imagen original y la máscara PIL y superponer
            resized_img = transforms.Resize((64, 64))(image)
            # activation_map[0] (Accede al primer y único mapa generado) .detach() (Devuelve una copia del tensor original que no está conectado a la gráfica de cálculo para evitar problemas)
            heat_map = overlay_mask(resized_img, to_pil_image(activation_map[0].detach(), mode='F'), alpha=0.5)
        except Exception as e:
            st.error(f"Error al generar el mapa Grad-CAM: {e}")
            heat_map = None

        #Mostrar el mapa de calor Grad-CAM
        if heat_map is not None:
            st.image(heat_map, caption="Mapa Grad-CAM: regiones sensibles al modelo", use_container_width=True)

            # Mostrar botón de descarga
            formato_gcam = st.selectbox("Formato de descarga Grad-CAM", ["PNG", "SVG"], key="formato_gcam")
            exportar_imagen_pil(heat_map, "Mapa Grad-CAM", formato_gcam)

    # Columa 3: Muestra el mapa de Saliencia
    with col3:
        # Título
        st.markdown("<h4 style='text-align: center;'>Mapa de Saliencia:</h4>", unsafe_allow_html=True)
        try:
            # Copiamos la imagen en forma de tensor para no modificar el original y lo separamos de  la gráfica de cálculo
            image_tensor = input_tensor.clone().detach()
            # Activamos el seguimiento de los gradientes
            image_tensor.requires_grad_()

            # Calculamos las salidas del modelo
            output_saliency = model(image_tensor)
            #Obtenemos el valor de salida [batch_size, num_classes], siendo el tamaño del lote de 1 (0) y la 'predicted _class' de 0 porque solo hay una neurona (clase)
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

        if saliency_img_resized is not None:
            #Mostramos el mapa de saliencia
            st.image(saliency_img_resized, caption="Mapa de saliencia: regiones sensibles al modelo", use_container_width=True)
            # Mostrar botón de descarga
            formato_sal = st.selectbox("Formato de descarga Saliencia", ["PNG", "SVG"], key="formato_sal")
            exportar_imagen_pil(saliency_img_resized, "Mapa de Saliencia", formato_sal)

# Divisor de las imágenes y el quesito de probabilidades
st.divider()

# === Estadísticas de la imagen ===
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
            # Crear gráfico de pastel
            fig, ax = plt.subplots(facecolor='#1e1e1e') # Color de fondo oscuro
            labels = ['Real', 'Fake']
            sizes = [probability, 1 - probability]
            colors = ['#00cc66', '#cc3333']
            # Explode automático si una parte es pequeña (<10%) -- Es decir, siempre
            explode = [0.1 if s < 0.1 else 0 for s in sizes]

            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.4f%%', startangle=90, colors=colors, explode=explode, textprops={'color': 'white', 'weight': 'bold', 'fontsize': 11})
            ax.axis('equal')  # Para que sea un círculo

            # Mostrar gráfico y caption
            st.pyplot(fig)
            st.caption("Distribución visual de la probabilidad predicha por el modelo")

        # No mostar gráfico:
        else:
            # Imagen real
            if probability >= 0.5 :
                st.markdown("Al seleccionar *\"Mostrar gráfico de distribución de probabilidad\"* en la barra lateral izquierda se creará un gráfico de probabilidad circular que expondrá en su parte superior la probabilidad de la clase *\"Fake\"* en la parte superior y la clase *\"Real\"* en su parte inferior")
                st.error(f"###### La probabilidad de que esta imagen sea generada sintéticamente ***(Fake)*** es del: ***{((1-probability)*100):.4f}%***")
                st.success(f"###### La probabilidad de que esta imagen sea real ***(Real)*** es del: ***{(probability*100):.4f}%***")  
            # Imagen falsa
            else:
                st.markdown("Al seleccionar *\"Mostrar gráfico de distribución de probabilidad\"* en la barra lateral izquierda se creará un gráfico de probabilidad circular que expondrá en su parte superior la probabilidad de la clase *\"Real\"* en la parte superior y la clase *\"Fake\"* en su parte inferior") 
                st.success(f"###### La **probabilidad** de que esta imagen sea real ***(Real)*** es del: ***{(probability*100):.4f}%***")
                st.error(f"###### La **probabilidad** de que esta imagen sea generada sintéticamente ***(Fake)*** es del: ***{((1-probability)*100):.4f}%***")
    
    # Columna 6: Mostar estadísticas y descarga del gráfico
    with col6:

        # Si se selecciona el checkbox:
        if mostrar_grafico:

            # Mostrar orden de imagen real (Fake-Real):
            if probability >= 0.5 :
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.error(f"###### La probabilidad de que esta imagen sea generada sintéticamente ***(Fake)*** es del: ***{((1-probability)*100):.4f}%***")
                st.success(f"###### La probabilidad de que esta imagen sea real ***(Real)*** es del: ***{(probability*100):.4f}%***")  
            
            # Mostar orden de imagen falsa (Real-Fake):
            else:    
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.success(f"###### La **probabilidad** de que esta imagen sea real ***(Real)*** es del: ***{(probability*100):.4f}%***")
                st.error(f"###### La **probabilidad** de que esta imagen sea generada sintéticamente ***(Fake)*** es del: ***{((1-probability)*100):.4f}%***")

            # Exportar gráfico circular en formato PNG y SVG

            # Selector de formato de exportación mediante un selectbox
            formato_exportacion = st.selectbox("Formato de exportación del gráfico", ["PNG", "SVG"])

            # Crear buffer en memoria
            buffer_grafico = io.BytesIO()

            # Guardar en el formato elegido
            fig.savefig(buffer_grafico, format=formato_exportacion.lower(), bbox_inches='tight', facecolor=fig.get_facecolor())
            buffer_grafico.seek(0)

            # Crear botón de descarga
            st.download_button(label=f"📥 Descargar gráfico como {formato_exportacion}", data=buffer_grafico, file_name=f"grafico_distribucion.{formato_exportacion.lower()}", mime="image/png" if formato_exportacion == "PNG" else "image/svg+xml")

            buffer_grafico.close()

            

# Footer
st.markdown("---")
st.markdown("TFG2")


