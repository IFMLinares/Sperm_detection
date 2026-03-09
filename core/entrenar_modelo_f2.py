import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import os
import matplotlib.pyplot as plt
import numpy as np
# --- ⚙️ CONFIGURACIÓN ---
DATASET_PATH = '../data/datasets/dataset_f2' 
OUTPUT_DIR = '../models/trained/clasificacion/experimento_1'
IMG_SIZE = 224
BATCH_SIZE = 16 # Batch más pequeño para dataset pequeño
EPOCHS = 50     # Subimos épocas porque usaremos Early Stopping

# Configuración para que la RTX 3080 gestione bien la memoria
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("🚀 GPU configurada y lista para la acción")
    except RuntimeError as e:
        print(e)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 📂 1. CARGA DE DATOS ---
# --- 📂 1. CARGA DE DATOS ---
print("📦 Cargando imágenes de dataset_f2...")
# Keras necesita un tamaño inicial. Usaremos el lado más largo (400) 
# para no perder detalle antes del padding.
LOAD_SIZE = 400 

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(LOAD_SIZE, LOAD_SIZE), # Tamaño temporal para carga
    batch_size=BATCH_SIZE,
    interpolation='bicubic'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(LOAD_SIZE, LOAD_SIZE),
    batch_size=BATCH_SIZE,
    interpolation='bicubic'
)

class_names = train_ds.class_names
print(f"Clases detectadas: {class_names}")

# --- 🛠️ 2. PRE-PROCESAMIENTO Y DATA AUGMENTATION ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
])

def prepare_image(image, label):
    # Redimensionamos con padding al tamaño final de MobileNet (224x224)
    # Esto asegura que la proporción del espermatozoide se mantenga intacta
    image = tf.image.resize_with_pad(image, IMG_SIZE, IMG_SIZE)
    return image, label

# Aplicamos padding y luego aumento de datos solo al entrenamiento
train_ds = train_ds.map(prepare_image).map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

val_ds = val_ds.map(prepare_image).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
# --- ⚖️ 3. MANEJO DE DESEQUILIBRIO (Class Weights) ---
# Le decimos a la IA que cada "Normal" vale por 3 o 4 "Anormales"
# Calculado: 120 / 33 approx 3.6
class_weight = {0: 1.0, 1: 3.6} # 0: Anormal, 1: Normal (ajustar según orden de class_names)
if class_names[0] == 'normal':
    class_weight = {0: 3.6, 1: 1.0}

# --- 🧠 4. MODELO ---
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False 

model = models.Sequential([
    layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Rescaling(1./127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3), # Más dropout para evitar que memorice las pocas fotos
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Early Stopping: Si el modelo deja de mejorar, se detiene solo
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# --- 🚀 5. ENTRENAMIENTO ---
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    class_weight=class_weight, # <--- APLICAMOS EL PESO AQUÍ
    callbacks=[callback]
)

# --- 💾 6. GUARDAR ---
model.save(os.path.join(OUTPUT_DIR, 'clasificador_morfologia_v1.keras'))
print("✅ Entrenamiento completado y modelo balanceado guardado.")

# --- 📊 7. GENERAR GRÁFICA DE RENDIMIENTO ---
print("📊 Generando reporte visual...")
plt.figure(figsize=(12, 5))

# Gráfica de Precisión (Accuracy)
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del Modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

# Gráfica de Pérdida (Loss)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Error')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'reporte_entrenamiento.png'))
print(f"✅ Gráfica guardada en: {OUTPUT_DIR}/reporte_entrenamiento.png")