Cat and Dog Image Classifier
Descripción

El Cat and Dog Image Classifier es un proyecto de clasificación de imágenes que utiliza un modelo de Red Neuronal Convolucional (CNN) para clasificar imágenes de gatos y perros. El modelo es entrenado utilizando un conjunto de datos de imágenes etiquetadas y puede predecir si una imagen dada contiene un gato o un perro. Este proyecto está implementado en Python utilizando TensorFlow y Keras, dos bibliotecas populares para el desarrollo de redes neuronales profundas.

Características

Clasificación binaria: El modelo clasifica las imágenes en dos categorías: gatos y perros.

Red Neuronal Convolucional (CNN): Se utiliza una red neuronal convolucional para procesar las imágenes y aprender patrones visuales importantes.

Entrenamiento y Evaluación: El modelo se entrena con un conjunto de datos etiquetado, y se evalúa utilizando métricas como precisión y recall.

Predicciones: El modelo puede predecir si una imagen contiene un gato o un perro con alta precisión.

Tecnologías utilizadas

Python: Para la implementación del modelo de red neuronal y el procesamiento de imágenes.

TensorFlow y Keras: Para crear, entrenar y evaluar el modelo de Red Neuronal Convolucional (CNN).

OpenCV: Para el procesamiento de imágenes y preprocesamiento antes de alimentarlas al modelo.

Matplotlib/Seaborn: Para la visualización de los resultados de la clasificación y la evaluación del modelo.

Cómo usar el proyecto

Clonar el repositorio
Si deseas clonar este proyecto, usa el siguiente comando:

git clone <repositorio_url>  


Instalar las dependencias
Instala las bibliotecas necesarias ejecutando:

pip install -r requirements.txt  


Entrenar el modelo

Para entrenar el modelo, ejecuta el archivo train_model.py. Este archivo se encargará de cargar el conjunto de datos, procesar las imágenes, y entrenar el modelo CNN.

Realizar predicciones

Para realizar predicciones sobre nuevas imágenes, ejecuta el archivo predict_image.py y proporciona la ruta a la imagen que deseas clasificar.

El modelo devolverá la predicción de si la imagen contiene un gato o un perro.

Evaluar el modelo

El archivo evaluate_model.py permite evaluar el rendimiento del modelo utilizando métricas como precisión, recall y F1-score para verificar la efectividad de la clasificación.

Instalación

Clona el repositorio y navega a la carpeta del proyecto.

Ejecuta pip install -r requirements.txt para instalar las dependencias necesarias.

Asegúrate de tener un entorno de Python 3.x para que las bibliotecas funcionen correctamente.

Contribuciones

Si deseas contribuir al proyecto, sigue estos pasos:

Haz un fork del repositorio.

Crea una rama para tu nueva funcionalidad o corrección de errores (git checkout -b nueva-funcionalidad).

Haz tus cambios y realiza un commit (git commit -am 'Añadir nueva funcionalidad').

Push a tu rama (git push origin nueva-funcionalidad).

Abre una pull request detallando los cambios realizados.

Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para obtener más detalles.
