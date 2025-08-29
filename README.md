🐶🐱 Cat and Dog Image Classifier

Clasificador de imágenes de gatos y perros utilizando redes neuronales convolucionales (CNN). Este proyecto muestra preprocesamiento de imágenes, entrenamiento de modelo con Keras/TensorFlow y evaluación de rendimiento.

📌 Resumen del Proyecto

Este proyecto entrena un modelo de Deep Learning para diferenciar entre gatos y perros a partir de un dataset de imágenes. Incluye pasos de:

Limpieza y organización de datos.

Generadores de imágenes para entrenamiento y validación.

Arquitectura CNN en TensorFlow/Keras.

Visualización de curvas de precisión y pérdida.

Evaluación final sobre datos de prueba.

🧰 Stack Tecnológico
Herramienta	Uso
Python 3.x	Lenguaje principal
TensorFlow / Keras	Definición y entrenamiento del modelo CNN
NumPy	Operaciones numéricas
Matplotlib	Gráficas de entrenamiento y resultados
scikit-learn	Métricas de evaluación
📁 Estructura de Archivos
Cat-and-Dog-Image-Classifier/
├── data/                      # Dataset (train/val/test)
├── models/                    # Modelos entrenados (H5)
├── notebooks/                 # Jupyter notebooks (opcional)
├── cat_dog_classifier.py      # Script principal de entrenamiento
├── requirements.txt           # Dependencias del proyecto
├── results.png                # Ejemplo de curvas de entrenamiento
└── README.md                  # Documentación

🚀 Cómo Ejecutar
1) Clonar el repositorio
git clone https://github.com/Emanuelgm777/Cat-and-Dog-Image-Classifier.git
cd Cat-and-Dog-Image-Classifier

2) Crear entorno virtual (opcional)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

3) Instalar dependencias
pip install -r requirements.txt

4) Entrenar el modelo
python cat_dog_classifier.py --epochs 20 --batch_size 32

5) Evaluar resultados

Se generarán gráficos de precisión y pérdida en results.png.
El modelo entrenado quedará guardado en la carpeta models/.

✅ Características Clave

Entrenamiento reproducible con TensorFlow/Keras.

Generadores de imágenes con ImageDataGenerator.

Visualización de métricas (accuracy, loss).

Código modular y extensible.

🧠 Qué Aprenderás

Cómo aplicar CNN en clasificación de imágenes.

Técnicas de data augmentation.

Uso de métricas de evaluación para visión por computadora.

🌍 Aplicaciones Reales

Clasificación de imágenes en sistemas inteligentes.

Modelos de visión computacional para mascotas y animales.

Base para proyectos de detección más complejos (p. ej., perros/gatos en tiempo real).

✍️ Autor

Emanuel González Michea
