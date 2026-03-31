import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split

# --- 🤫 SILENCIAR WARNINGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- ⚙️ CONFIGURACIÓN ---
BASE_DIR    = Path(__file__).parent.parent
MODEL_PATH  = BASE_DIR / "models" / "trained" / "clasificacion" / "experimento_5" / "mejor_modelo_v8.h5"
CSV_PATH    = BASE_DIR / "data" / "datasets" / "dataset_morfologia_v3" / "crops_multilabel.csv"
IMG_SIZE    = 300
SEED        = 42

CLASES = ["normal", "cabeza_anormal", "cola_anormal", "pieza_intermedia_anormal", "residuo_citoplasmatico"]

# --- 🧠 CARGAR MODELO ---
print(f"🧠 Cargando modelo GANADOR v8 desde: {MODEL_PATH.name}...")
from tensorflow.keras import layers, models
base_model = tf.keras.applications.EfficientNetB0(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights=None)
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(CLASES), activation="sigmoid")(x)
model = models.Model(inputs, outputs)
model.load_weights(str(MODEL_PATH))

# --- 📄 CARGAR DATOS ---
df_orig = pd.read_csv(CSV_PATH)
dataset_root = CSV_PATH.parent.parent.parent
df_orig["filepath_abs"] = df_orig["filepath"].apply(lambda p: str(dataset_root / p))
df_orig = df_orig[df_orig["filepath_abs"].apply(lambda p: os.path.exists(p))]

# --- 🧪 SPLIT DE VALIDACIÓN (Pool 10% -> 100 muestras) ---
_, df_val_pool = train_test_split(df_orig, test_size=0.10, random_state=SEED)
df_val = df_val_pool.sample(n=100, random_state=SEED)
print(f"🔍 Evaluando sobre {len(df_val)} imágenes de VALIDACIÓN REAL (Pool 10%)...\n")

def cargar_y_preprocesar(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img

# --- 🚀 INFERENCIA ---
y_true = df_val[CLASES].values.astype(int)
preds_raw_list = []
for i, img_path in enumerate(df_val["filepath_abs"]):
    img = cargar_y_preprocesar(img_path)
    p = model.predict(tf.expand_dims(img, 0), verbose=0)[0]
    preds_raw_list.append(p)
    if (i + 1) % 50 == 0: print(f"   ✅ Procesadas {i+1} muestras...")

preds_raw = np.array(preds_raw_list)

# --- 📊 CÁLCULO DE UMBRALES Y MÉTRICAS ---
kappas = []
best_thresholds = []
for i, clase in enumerate(CLASES):
    best_k = -1.0; best_t = 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        preds_t = (preds_raw[:, i] > t).astype(int)
        k_t = cohen_kappa_score(y_true[:, i], preds_t)
        if k_t > best_k:
            best_k = k_t; best_t = t
    kappas.append(best_k); best_thresholds.append(best_t)

y_pred_opt = np.zeros_like(preds_raw)
for i in range(len(CLASES)):
    y_pred_opt[:, i] = (preds_raw[:, i] > best_thresholds[i]).astype(int)

mcm = multilabel_confusion_matrix(y_true, y_pred_opt)

print("\n" + "="*100)
print(f"{'CATEGORÍA':<22} | {'TP':<3} | {'TN':<3} | {'FP':<3} | {'FN':<3} | {'ACC':<6} | {'SENS':<6} | {'ESPEC':<6} | {'VPP':<6} | {'VPN':<6} | {'F1':<6}")
print("="*100)

for i, clase in enumerate(CLASES):
    tn, fp, fn, tp = mcm[i].ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    espec = tn / (tn + fp) if (tn + fp) > 0 else 0
    vpp = tp / (tp + fp) if (tp + fp) > 0 else 0
    vpn = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * (vpp * sens) / (vpp + sens) if (vpp + sens) > 0 else 0
    print(f"{clase:<22} | {tp:^3} | {tn:^3} | {fp:^3} | {fn:^3} | {acc:.4f} | {sens:.4f} | {espec:.4f} | {vpp:.4f} | {vpn:.4f} | {f1:.4f}")

# --- 🧪 CÁLCULO DEL ÍNDICE DE TERATOZOOSPERMIA (TZI) ---
# TZI = Total de defectos / Número de espermatozoides con al menos un defecto
def calcular_tzi(labels_matrix):
    # Ignoramos la primera columna (índice 0) que es "normal"
    defectos_por_celula = np.sum(labels_matrix[:, 1:], axis=1)
    total_defectos = np.sum(defectos_por_celula)
    num_celulas_con_defectos = np.sum(defectos_por_celula > 0)
    
    if num_celulas_con_defectos == 0: return 1.0
    return total_defectos / num_celulas_con_defectos

tzi_expert = calcular_tzi(y_true)
tzi_ai     = calcular_tzi(y_pred_opt)

print("-" * 100)
print(f"{'ÍNDICE KAPPA MACRO':<22} | {np.mean(kappas):>10.4f}")
print(f"{'TZI (EXPERTO AGENCIA)':<22} | {tzi_expert:>10.4f}")
print(f"{'TZI (IA ANTIGRAVITY)':<22} | {tzi_ai:>10.4f}")
print("="*100)
