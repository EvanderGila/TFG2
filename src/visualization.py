"""Este módulo contiene funciones para la visualización de resultados"""
# librerías estándar
import io

# Librerías externas
import streamlit as st
import matplotlib.pyplot as plt


# === VISUALIZACIÓN ===

# Mostrar la confianza del modelo en probabilidad 
def display_prediction(probability: float | None):
    #"""Muestra el resultado de la predicción."""
    if probability is not None:
        # Clasificación de la imagen y confianza según probabilidad
        if probability >= 0.5:
            prediction = "Esta imagen es **real**"
            confidence = probability * 100 # Usamos la propia probabilidad del modelo (1 = Real)
        else:
            prediction = "Esta imagen está **generada sintéticamente (FAKE)**"
            confidence = (1 - probability) * 100 # Invertimos la probabilidad del modelo (0 = Fake)

    return prediction, confidence

# Mostrar el gráfico de probabilidad (quesito)
def display_probability_chart(probability: float):
    """Muestra un gráfico de pastel con la distribución de probabilidad"""
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

    return fig

# Mostrar texto con probabilidades extendido (No mostrar gráfico)
def display_probability_text_extended(probability: float):
    # """Muestra el texto con las probabilidades de real y fake cuando no está activo el gráfico"""
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


def display_probability_text(probability: float):
    # """Muestra el texto con las probabilidades de real y fake cuando está activo el gráfico"""
    # Mostrar orden de imagen real (Fake-Real):
    if probability >= 0.5 :
        st.error(f"###### La probabilidad de que esta imagen sea generada sintéticamente ***(Fake)*** es del: ***{((1-probability)*100):.4f}%***")
        st.success(f"###### La probabilidad de que esta imagen sea real ***(Real)*** es del: ***{(probability*100):.4f}%***")  
            
    # Mostar orden de imagen falsa (Real-Fake):
    else:    
        st.success(f"###### La **probabilidad** de que esta imagen sea real ***(Real)*** es del: ***{(probability*100):.4f}%***")
        st.error(f"###### La **probabilidad** de que esta imagen sea generada sintéticamente ***(Fake)*** es del: ***{((1-probability)*100):.4f}%***")

# === DESCARGAS ===

# Descargar los mapas de Grad-CAM, Saliency y LIME
def export_imagen_pil(imagen_pil, nombre_archivo, formato):
    # """Permite exportar y descargar imagenes pil"""

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

# Exportar el gráfico de quesito
def export_graph(fig):
    # """Permite la exportación del gráfico de quesito"""

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

# Seleccionar 'alpha' de Grad-CAM
def alpha_gradcam():
    # Título
    st.sidebar.subheader("Opciones de Explicación Grad-CAM")
    # """Permite la selección del valor de opacidad del mapa de calor de Grad-CAM"""
    alpha = st.sidebar.slider("Transparencia Grad-CAM", 0.0, 1.0, 0.5) # Min, Max, Valor por defecto
    return alpha

# Mostar los detalles del modelo
def show_model_details(model_choice):
    # """Permite mostrar los detalles del modelo seleccionado"""
    if model_choice == "CNN_3C":
        st.write("""
            #### CNN de 3 Capas Convolucionales
            Esta arquitectura consiste en tres capas convolucionales seguidas de capas de normalización por lotes, funciones de activación ReLU y capas de max-pooling.
            La función de activación es la sigmoid.
            """)
        st.markdown("Número de capas convolucionales: **3**")
        st.markdown("Funciones de activación: **ReLU**")
        st.markdown("Pooling: **Max Pooling**")
        st.markdown("Número de pooling: **3**")
        st.markdown("Dropout: **0.5**")

    elif model_choice == "CNN_4C":
        st.write("""
            #### CNN de 4 Capas Convolucionales
            Esta arquitectura consiste en cuatro capas convolucionales seguidas de capas de normalización por lotes, funciones de activación ReLU y capas de max-pooling.
            La función de activación es la sigmoid.
            """)
        st.markdown("Número de capas convolucionales: **4**")
        st.markdown("Funciones de activación: **ReLU**")
        st.markdown("Pooling: **Max Pooling**")
        st.markdown("Número de pooling: **4**")
        st.markdown("Dropout: **0.5**")
    else:
        st.info("Selecciona una arquitectura para ver sus detalles.")

# Variar opciones de  LIME 
def lime_options():
    # ''' Función que permite variar opciones de visualización de LIME'''
    st.sidebar.subheader("Opciones de Explicación LIME")
    
    # Opción para alterar hide_rest
    hide_rest_selected = st.sidebar.checkbox(
        "Ocultar el resto de la imagen (solo mostrar regiones relevantes)",
        value=False, # Por defecto, no ocultamos el resto
        key="hide_rest_lime"
    )

    hide_color_value = None

    if hide_rest_selected:

        # Opción para cambiar el color de hide_color
        hide_color_option = st.sidebar.radio(
            "Color de las áreas ocultas",
            ["Negro", "Gris", "Blanco"],
            key="hide_color_lime"
        )

        hide_color_value = 0 # Valor por defecto (Negro)
        if hide_color_option == "Gris":
            hide_color_value = 128
        elif hide_color_option == "Blanco":
            hide_color_value = 255

    return hide_rest_selected, hide_color_value