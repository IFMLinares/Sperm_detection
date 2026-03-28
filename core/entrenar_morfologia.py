"""
entrenar_morfologia.py
==============================
Estrategia Ganadora (v8): EfficientNetB0 + Focal Loss + Oversampling Físico.
Objetivo: Clasificación multietiqueta de morfología espermática.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

# --- ⚙️ CONFIGURACIÓN ---
BASE_DIR    = Path(__file__).parent.parent
CSV_PATH    = BASE_DIR / "data" / "datasets" / "dataset_morfologia_v3" / "crops_multilabel.csv"
OUTPUT_DIR  = BASE_DIR / "models" / "trained" / "clasificacion" / "experimento_5" # Folder del Experimento Ganador
IMG_SIZE    = 300 # Aumentamos resolución para más detalle
BATCH_SIZE  = 16 
EPOCHS      = 60
SEED        = 42

CLASES = ["normal", "cabeza_anormal", "cola_anormal",
          "pieza_intermedia_anormal", "residuo_citoplasmatico"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 🧪 1. BALANCEO FÍSICO (OVERSAMPLING) ---
df_orig = pd.read_csv(CSV_PATH)
dataset_root = CSV_PATH.parent.parent.parent
df_orig["filepath_abs"] = df_orig["filepath"].apply(lambda p: str(dataset_root / p))
df_orig = df_orig[df_orig["filepath_abs"].apply(lambda p: os.path.exists(p))]

# --- 🏗️ 2. DATA PIPELINE (Split ANTES de Oversampling para evitar fuga de datos) ---
df_train_orig, df_val = train_test_split(df_orig, test_size=0.10, random_state=SEED)

# Aplicar Oversampling SOLO al set de entrenamiento
df_normal_t = df_train_orig[df_train_orig["normal"] == 1]
df_anormal_t = df_train_orig[df_train_orig["normal"] == 0]

df_train = pd.concat([pd.concat([df_normal_t] * 12, ignore_index=True), df_anormal_t], ignore_index=True).sample(frac=1.0, random_state=SEED)

print(f"⚖️ Dataset Train: {len(df_train)} (Normales: {len(df_train[df_train['normal']==1])})")
print(f"⚖️ Dataset Val: {len(df_val)} (Normales: {len(df_val[df_val['normal']==1])})")

def cargar_y_preprocesar(path, labels):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img, labels

def aumentar(img, labels):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    # Rotación aleatoria completa (espermatozoides no tienen arriba/abajo fijo)
    img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return img, labels

train_ds = tf.data.Dataset.from_tensor_slices((df_train["filepath_abs"].values, df_train[CLASES].values.astype(np.float32)))
train_ds = train_ds.shuffle(2000).map(cargar_y_preprocesar).map(aumentar).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((df_val["filepath_abs"].values, df_val[CLASES].values.astype(np.float32)))
val_ds = val_ds.map(cargar_y_preprocesar).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- 🏗️ 3. FOCAL LOSS (Para desbalance y ejemplos difíciles) ---
class MultiLabelFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name="focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        
        # Focal Loss para clasificación binaria/multietiqueta
        # Loss = -alpha * (1-p)^gamma * y_true * log(p) - (1-alpha) * p^gamma * (1-y_true) * log(1-p)
        
        # Parte positiva
        pos_loss = -y_true * tf.math.pow(1.0 - y_pred, self.gamma) * tf.math.log(y_pred)
        # Parte negativa
        neg_loss = -(1.0 - y_true) * tf.math.pow(y_pred, self.gamma) * tf.math.log(1.0 - y_pred)
        
        loss = self.alpha * pos_loss + (1.0 - self.alpha) * neg_loss
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config

# --- 🏗️ 4. CALLBACKS ---
class LogCleaning(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k in list(logs.keys()):
                v = logs[k]
                try:
                    if hasattr(v, 'numpy'): v = v.numpy()
                    if isinstance(v, (np.ndarray, np.generic)):
                        logs[k] = float(v) if v.size == 1 else float(np.mean(v))
                    else:
                        logs[k] = float(v)
                except:
                    del logs[k]

tf.keras.utils.get_custom_objects().update({'MultiLabelFocalLoss': MultiLabelFocalLoss})

# --- 🏗️ 5. MODELO ---
base_model = tf.keras.applications.EfficientNetB0(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base_model.trainable = False 

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation="relu")(x) # Capa más ancha para 300px
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(CLASES), activation="sigmoid")(x)

model = models.Model(inputs, outputs)

# --- 🚀 6. FASE 1: CALENTAMIENTO ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=MultiLabelFocalLoss(),
    metrics=[tf.keras.metrics.AUC(multi_label=True, name="auc"), 'accuracy']
)

print("\n🚀 Fase 1: Calentamiento (IMG_SIZE=300)...")
model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=[LogCleaning()])

# --- 🚀 7. FASE 2: FINE-TUNING TOTAL ---
print("\n🚀 Fase 2: Fine-tuning con Focal Loss...")
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=MultiLabelFocalLoss(),
    metrics=[tf.keras.metrics.AUC(multi_label=True, name="auc"), 'accuracy']
)

callbacks = [
    LogCleaning(),
    tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=10, restore_best_weights=True, mode="max"),
    tf.keras.callbacks.ModelCheckpoint(filepath=str(OUTPUT_DIR / "mejor_modelo_v8.h5"), monitor="val_auc", save_best_only=True, mode="max", save_weights_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
]

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# El guardado final del modelo completo suele fallar por errores de serialización JSON en Keras/TF 2.10.
# Sin embargo, el ModelCheckpoint con 'save_weights_only=True' ya guardó los mejores pesos en 'mejor_modelo_v8.h5'.
print(f"\n✅ Entrenamiento finalizado.")
print(f"📍 Los mejores pesos están en: {OUTPUT_DIR}/mejor_modelo_v8.h5")
print(f"🔍 Puedes ejecutar ahora: python core/calcular_kappa.py")
