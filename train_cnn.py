import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

BASE_DIR = 'cats_and_dogs'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR   = os.path.join(BASE_DIR, 'validation')
TEST_DIR  = BASE_DIR  # para usar classes=['test']

IMG_HEIGHT = 150
IMG_WIDTH  = 150
BATCH_SIZE = 32
EPOCHS     = 5  # sube a 15-20 si quieres más accuracy

# Generators (rescale)
train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data_gen = validation_image_generator.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Truco para test sin etiquetas ni subdirs: classes=['test'] y shuffle=False
test_data_gen = test_image_generator.flow_from_directory(
    TEST_DIR,
    classes=['test'],
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False
)

# Data augmentation para mejorar generalización (rápido)
train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_data_gen = train_aug.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Modelo CNN simple y efectivo
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

steps_per_epoch  = max(1, train_data_gen.samples // BATCH_SIZE)
validation_steps = max(1, val_data_gen.samples // BATCH_SIZE)

print("\n==> Entrenando...")
history = model.fit(
    train_data_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=validation_steps,
    verbose=2
)

print("\n==> Evaluando en validación completa...")
val_loss, val_acc = model.evaluate(val_data_gen, verbose=0)
print(f"Validation accuracy: {val_acc:.4f}")

print("\n==> Prediciendo sobre test (50 imágenes, orden estable)...")
probs = model.predict(
    test_data_gen,
    steps=int(np.ceil(test_data_gen.samples / BATCH_SIZE)),
    verbose=0
).ravel()

# Guardamos predicciones (0=cat, 1=dog) y probabilidades
pred_labels = (probs >= 0.5).astype(int)
np.savetxt("test_predictions_labels.txt", pred_labels, fmt="%d")
np.savetxt("test_predictions_probs.txt", probs, fmt="%.6f")

# Guardamos el modelo por si quieres reutilizar
model.save("cats_dogs_model.h5")
print("\nListo. Archivos generados: cats_dogs_model.h5, test_predictions_labels.txt, test_predictions_probs.txt")
