"""Este archivo contiene ejemplos para aprender a usar Streamlit."""

import streamlit as st
import time
import numpy as np
import pandas as pd
from PIL import Image

# Título
st.title("🔧 Taller interactivo de Streamlit")
st.write("Este archivo es solo para aprender cómo funciona Streamlit con ejemplos prácticos.")

# 🧪 1. Texto y entrada de datos
st.header("1️⃣ Entradas de texto")
nombre = st.text_input("¿Cómo te llamas?")
if nombre:
    st.success(f"¡Hola, {nombre}!")

# 🧮 2. Contadores con session_state
st.header("2️⃣ Uso de session_state para mantener estado")

if "contador" not in st.session_state:
    st.session_state.contador = 0

if st.button("➕ Aumentar contador"):
    st.session_state.contador += 1

st.info(f"Contador actual: {st.session_state.contador}")

# 🎛 3. Widgets interactivos
st.header("3️⃣ Sliders y checkboxes")
valor_slider = st.slider("Selecciona un valor entre 0 y 100", 0, 100, 50)
mostrar = st.checkbox("¿Mostrar valor?")
if mostrar:
    st.write(f"Seleccionaste: {valor_slider}")

# 🧠 4. Selectbox y lógica condicional
st.header("4️⃣ Selectbox")
opcion = st.selectbox("Elige un modelo de ejemplo", [
                      "Modelo A", "Modelo B", "Modelo C"])
st.write(f"Has elegido: {opcion}")

# 📈 5. Gráficos dinámicos
st.header("5️⃣ Gráficos")

st.subheader("📊 Línea aleatoria")
datos = pd.DataFrame(np.random.randn(50, 3), columns=["A", "B", "C"])
st.line_chart(datos)

st.subheader("🗺 Mapa aleatorio")
mapa = pd.DataFrame(
    np.random.randn(100, 2) / [50, 50] +
    [40.4168, -3.7038],  # Madrid como centro
    columns=['lat', 'lon'])
st.map(mapa)

# 🖼 6. Imagen y subida
st.header("6️⃣ Subida de imagen")
imagen = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])
if imagen:
    img = Image.open(imagen)
    st.image(img, caption="Imagen subida", use_column_width=True)

# ⏱ 7. Barras de progreso
st.header("7️⃣ Barra de progreso simulada")
if st.button("Simular proceso"):
    progreso = st.progress(0)
    for i in range(100):
        time.sleep(0.02)
        progreso.progress(i + 1)
    st.success("¡Proceso completo!")

# 💬 8. Mensajes de estado
st.header("8️⃣ Mensajes informativos")
st.success("✅ Esto es un mensaje de éxito.")
st.warning("⚠️ Esto es una advertencia.")
st.error("❌ Esto es un error.")
st.info("ℹ️ Esto es solo información.")

# 📚 9. Código y ayuda
st.header("9️⃣ Mostrar código")
codigo = '''def saludar(nombre):
    return f"Hola, {nombre}!"
'''
st.code(codigo, language="python")

with st.expander("💡 ¿Qué es Streamlit?"):
    st.markdown("""
        **Streamlit** es una librería de Python para crear interfaces web de forma rápida y sencilla.  
        Ideal para dashboards de datos, prototipos de ML o aplicaciones interactivas.
    """)

# Fin
st.markdown("---")
st.write("Fin del tutorial")
