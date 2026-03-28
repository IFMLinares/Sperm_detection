import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
CSV_PATH = BASE_DIR / "data" / "datasets" / "dataset_morfologia_v3" / "crops_multilabel.csv"
MODEL_PATH = BASE_DIR / "models" / "trained" / "clasificacion" / "experimento_3" / "mejor_modelo_v5.keras"

CLASES = ["normal", "cabeza_anormal", "cola_anormal",
          "pieza_intermedia_anormal", "residuo_citoplasmatico"]

print("--- ANALISIS DE DATASET ---")
df = pd.read_csv(CSV_PATH)
print(f"Total filas: {len(df)}")
for c in CLASES:
    count = df[c].sum()
    print(f"Clase {c:25}: {count} ({count/len(df)*100:.2f}%)")

print("\n--- ANALISIS DE PREDICCIONES (v5) ---")
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

# Tomar 50 imágenes al azar del test set
img_dir = BASE_DIR / "data" / "roboflow_dataset" / "test" / "images"
imgs = list(img_dir.glob("*.jpg"))[:50]

preds = []
for p in imgs:
    img = tf.io.read_file(str(p))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    preds.append(model.predict(tf.expand_dims(img, 0), verbose=0)[0])

preds = np.array(preds)
print(f"\nRango de probabilidades (Min/Max/Mean):")
for i, c in enumerate(CLASES):
    p_c = preds[:, i]
    print(f"{c:25}: {p_c.min():.4f} / {p_c.max():.4f} / {p_c.mean():.4f}")

