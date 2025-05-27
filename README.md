# ***(((CAMBIAR POR EL TÍTULO DEL PROYECTO!!!)))***

## *Evander Gila Reques, TFG de Ing. de Tecnologías para la Sociedad de la Información*

#### { La URL para lanzar la aplicación en Streamlit Cloud es: https://fakeimagedetectortfg.streamlit.app/ }

Este proyecto se fundamenta en el creciente auge de las IAs generativas en un mundo global cada vez más interconectado, utilizadas en ocasiones para la creación de *"fake news"*. Actualmente la calidad de estas imágenes generadas sintéticamente ha avanzado a pasos agigantados, siendo muy difíciles de reconocer y distinguir de una imagen real.
(((AÑADIR MÁS INTRODUCCIÓN)))

### *Dataset*

Para el entrenamiento de los modelos disponibles en la aplicación, se ha usado el *dataset "CIFAKE: Real and AI-Generated Synthetic Images"* (https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images), el cual está diseñado específicamente para abordar este tipo de problemas (imágenes reales  vs fake), conteniendo 120K imágenes, siendo 60K reales y 60K generadas sintéticamente. Las imágenes destinadas al entrenamiento serían 100K y a test serían 20K. Las imágenes reales proceden del *dataset CIFAR-10*, con imágenes a color de 32x32 píxeles distribuidas en 10 clases. Las imágenes sintéticas se generaron mediante *Stable Diffusion v1.4*. Por tanto, este *dataset* ofrece una báse sólida para el entrenamiento de los modelos usados en este proyecto.

### *Modelos*

Los modelos disponibles se encuentran disponibles en el repositorio de GitHub: https://github.com/EvanderGila/TFG1 , repositorio en el cual se encuentra su proceso de entrenamiento, validación y test. En este se encuentran varios modelos pertenecientes a varias arquitecturas diseñadas y entrenados variando hiperparámetros tales como el "learning rate" o el uso de "Data augmentantion" para su entrenamiento.
Los modelos seleccionados son el modelo de tres capas "Model1_3C_3_NO.pth" y el modelo de cuatro capas "Model1_4C_2_DA.pth", renombrados como "Model3C F" y "Model4C C" en el  documento oficial.
El modelo "Model3C F", perteneciente a la arquitectura de tres capas, está entrenado a 20 epochs, con un learning rate de 0.0002, sin Data augmentation.
El modelo "Model4C C", perteneciente a la arquitectura de cuatro capas, está entrenado a 20 epochs, con un learning rate de 0.0001, con Data augmentation.


((((CONTINUAR)))))


