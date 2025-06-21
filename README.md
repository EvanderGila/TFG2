# ***(((CAMBIAR POR EL TÍTULO DEL PROYECTO!!!)))***

## *Evander Gila Reques, TFG de Ing. de Tecnologías para la Sociedad de la Información*

#### { La URL para lanzar la aplicación en Streamlit Cloud es: https://fakeimagedetectortfg.streamlit.app/ }

###  *Abstract/resumen:*

En los últimos años, se ha experimentado un auge exponencial de modelos de generación de imágenes, las cuales son a su vez cada vez más realistas, favoreciendo la equivocación de los usuarios y por tanto, creando un creciente ecosistema de desinformación. Por otra parte, las herramientas de detección y discriminación de este contenido es escaso y no está estandarizado para el público general. Todo esto genera una disparidad entre el avance de estos dos campos. Ante este contexto, el presente trabajo tiene como objetivo principal el desarrollo de una herramienta sencilla y práctica que permita luchar contra la incertidumbre que estas imágenes generan, utilizando modelos de inteligencia artificial que permitan discernir cuáles de estas están generadas sintéticamente, aplicando técnicas de explicación de los resultados obtenidos con el fin de entenderlos. Además, esta herramienta estará orientada a actualizarse de una forma sencilla que permita seguir el ritmo a los nuevos modelos generativos.

### *Dataset*

Para el entrenamiento de los modelos disponibles en la aplicación, se ha usado el *dataset "CIFAKE: Real and AI-Generated Synthetic Images"* (https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images), el cual está diseñado específicamente para abordar este tipo de problemas (imágenes reales  vs fake), conteniendo 120K imágenes, siendo 60K reales y 60K generadas sintéticamente. Las imágenes destinadas al entrenamiento serían 100K y a test serían 20K. Las imágenes reales proceden del *dataset CIFAR-10*, con imágenes a color de 32x32 píxeles distribuidas en 10 clases. Las imágenes sintéticas se generaron mediante *Stable Diffusion v1.4*. Por tanto, este *dataset* ofrece una base sólida para el entrenamiento de los modelos usados en este proyecto.

### *Modelos*

Los modelos utilizados se encuentran disponibles en el repositorio de GitHub: https://github.com/EvanderGila/TFG1 , repositorio en el cual se encuentra su proceso de entrenamiento, validación y test. En este se encuentran varios modelos pertenecientes a varias arquitecturas diseñadas y entrenados variando hiperparámetros tales como el *"learning rate"* o el uso de *"Data augmentantion"* para su entrenamiento.
Los modelos seleccionados son el modelo de tres capas *"Model1_3C_3_NO.pth"* y el modelo de cuatro capas "Model1_4C_2_DA.pth", renombrados como "Model3C F" y "Model4C C" en el  documento oficial.
El modelo "Model3C F", perteneciente a la arquitectura de tres capas, está entrenado a 20 epochs, con un learning rate de 0.0002, sin Data augmentation.
El modelo "Model4C C", perteneciente a la arquitectura de cuatro capas, está entrenado a 20 epochs, con un learning rate de 0.0001, con Data augmentation.


### *Streamlit*

Esta aplicación ha sido desarrollada con [Streamlit](https://streamlit.io), un *framework* en *Python* que permite crear interfaces web de forma rápida y sencilla, ideal para prototipado y despliegue de modelos de *machine learning*.
Funcionalidades principales aplicadas:
- Carga de imágenes por parte del usuario.
- Selección de modelo desde menús desplegables.
- Predicciones mostradas con texto y probabilidades.
- Visualización de mapas de explicabilidad (XAI).
- Interfaz interactiva con controles como sliders, botones y menús.

La app puede ejecutarse localmente o desplegarse en la nube mediante plataformas como *Streamlit Cloud* o *Heroku*.

### *Técnicas XAI utilizadas*

Con el objetivo de interpretar el comportamiento de los modelos de clasificación utilizados en la aplicación, se han integrado diversas técnicas de explicabilidad *(eXplainable AI - XAI)*, que permiten visualizar qué regiones de la imagen han influido más en la decisión del modelo.
Las técnicas implementadas son:

#### *Grad-CAM (Gradient-weighted Class Activation Mapping)*

*Grad-CAM* permite visualizar las zonas activadas de una imagen en función de los gradientes que fluyen hacia las capas finales de una red convolucional. Se utiliza para generar mapas de calor que resaltan las regiones más relevantes para una predicción concreta. En la aplicación los mapas *Grad-CAM* se generan automáticamente tras la predicción y se superponen sobre la imagen original para facilitar la interpretación visual, además, se pueden descargar  en formato *png* o *svg*.
Para su implementación se ha usado la librería ***torchcam***

#### *Mapas de saliencia (Saliency maps)*

Los mapas de saliencia destacan los píxeles que más afectan la salida del modelo, calculando el gradiente de la clase predicha respecto a cada píxel de la imagen. Esta técnica ofrece una visión más detallada y de bajo nivel que *Grad-CAM*.  En la aplicación se muestra un mapa en escala de grises que refleja la sensibilidad de cada píxel frente al resultado. La implementación en este caso ha sido propia, a partir de los gradientes del modelo utilizando ***Pytorch***.

#### *LIME (Local Interpretable Model-agnostic Explanations)*

*LIME* genera explicaciones locales para cada predicción al crear múltiples versiones perturbadas de la imagen original y entrenar un modelo interpretable sobre esos datos. De esta forma, se puede estimar qué regiones contribuyen más a una clasificación específica. En la aplicación se generan superpixeles relevantes para cada clase predicha (fake o real), facilitando una interpretación intuitiva y modelo-agnóstica  (la explicación no conoce el modelo usado). Para su implementación se ha usado la librería ***lime***

### *Módulos*

La aplicación ha sido diseñada con un enfoque modular, dividiendo el código en archivos separados según su funcionalidad. Esta organización favorece la claridad, facilita el mantenimiento y permite ampliar el sistema de forma ordenada y eficiente. 
El archivo principal de la aplicación es: ***app.py***, que actúa como punto de entrada para la interfaz desarrollada con *Streamlit*.

A continuación, se describen los módulos de la aplicación:

- ***preprocess.py***:  Encargado de cargar las imágenes proporcionadas por el usuario, convertirlas a formato RGB y aplicar el preprocesamiento necesario antes de su análisis por parte del modelo.
- ***models.py***: Define las arquitecturas de redes convolucionales disponibles en la aplicación, utilizadas para realizar las predicciones.
- ***model_loading.py***: Gestiona la carga del modelo seleccionado y el procesamiento de la imagen de entrada, devolviendo los resultados listos para su visualización.
- ***gradcam_utils.py***: Contiene las funciones auxiliares para generar mapas de activación Grad-CAM a partir de los gradientes del modelo.
- ***explanation.py***: Agrupa todas las funciones relacionadas con la generación de explicaciones visuales mediante técnicas *XAI* como *Grad-CAM*, mapas de saliencia y *LIME*.
- ***visualization.py***: Maneja la representación gráfica y textual de los resultados, incluyendo mapas de calor, saliencia y superpixeles explicativos.


### *Dependencias*

La aplicación está construida sobre un conjunto de librerías ampliamente utilizadas en proyectos de visión por computador e interfaces interactivas con *Python*. A continuación se enumeran las principales dependencias utilizadas:

- ***Streamlit***: Para el desarrollo de la interfaz web interactiva.

- ***PyTorch*** y ***Torchvision***: Utilizados para definir, entrenar y utilizar modelos de *deep learning*.

- ***TorchCAM***: Empleado para generar mapas de activación *Grad-CAM* a partir de los modelos de *PyTorch*.

- ***Pillow***: Para el manejo y procesamiento básico de imágenes.

- ***Matplotlib***: Para la visualización de resultados y generación de gráficos.

- ***LIME***: Para la generación de explicaciones locales e interpretables de las predicciones del modelo.

- ***Scikit-Image***: Para operaciones adicionales de procesamiento de imágenes, como la segmentación en superpixeles.

Estas dependencias pueden instalarse fácilmente mediante pip utilizando el archivo ***requirements.txt*** incluido en el repositorio.


###### Este proyecto se basa en mi  primer proyecto de fin de grado y constituye un punto de partida para la lucha contra la desinformación.

