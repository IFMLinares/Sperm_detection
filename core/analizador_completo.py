import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import os
import glob
from pathlib import Path

# --- ⚙️ CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).parent.parent

MODELO_YOLO = str(BASE_DIR / "models/trained/runs/detect/trained_sperm_model/weights/best.pt")
MODELO_KERAS = str(BASE_DIR / "models/trained/clasificacion/experimento_5/mejor_modelo_v8.h5")

DIR_PRUEBAS = str(BASE_DIR / "data/raw/muestras_variadas")
DIR_RESULTADOS = str(BASE_DIR / "docs/pruebas/muestras_variadas_reporte")

IMG_SIZE = 300
UMBRAL_MORFOLOGIA = 0.5 

# Forzar salida UTF-8 para evitar errores en Windows
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# --- 🛠️ FUNCIONES PERSONALIZADAS (Para cargar el modelo v3) ---
def weighted_binary_crossentropy(y_true, y_pred):
    return tf.keras.backend.binary_crossentropy(y_true, y_pred)

# --- 🧠 1. CARGAR INTELIGENCIA ---
print("🧠 Cargando inteligencia multi-label en la GPU...")

# Configurar memoria GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

if not os.path.exists(MODELO_YOLO):
    print(f"⚠️ No se encontró el modelo YOLO en: {MODELO_YOLO}")
    exit()

detector = YOLO(MODELO_YOLO)

if os.path.exists(MODELO_KERAS):
    # El modelo v8 usa pesos (.h5) y requiere definir la arquitectura (EfficientNetB0)
    print("🧠 Construyendo arquitectura EfficientNetB0 para v8...")
    from tensorflow.keras import layers, models
    
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), 
        include_top=False, 
        weights=None
    )
    
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(5, activation="sigmoid")(x)
    
    clasificador = models.Model(inputs, outputs)
    clasificador.load_weights(MODELO_KERAS)
    
    CLASES = ["normal", "cabeza_anormal", "cola_anormal", "pieza_intermedia_anormal", "residuo_citoplasmatico"]
else:
    print(f"❌ No se encontró el modelo Keras v8 en: {MODELO_KERAS}")
    exit()

# --- 🖼️ 2. PROCESAR TODAS LAS IMÁGENES ---
imagenes = glob.glob(os.path.join(DIR_PRUEBAS, "*.*"))

if not imagenes:
    print(f"⚠️ No se encontraron imágenes en {DIR_PRUEBAS}")
    exit()

print(f"🚀 Iniciando análisis morfológico detallado de {len(imagenes)} muestras...")

for ruta_img in imagenes:
    nombre_archivo = os.path.basename(ruta_img)
    print(f"🔍 Analizando: {nombre_archivo}")
    
    img = cv2.imread(ruta_img)
    if img is None: continue
    img_anotada = img.copy()
    
    # A) Detección Fase 1 (YOLO)
    results = detector.predict(img, conf=0.2, verbose=False)
    
    # Estadísticas globales de la imagen
    stats = {clase: 0 for clase in CLASES}
    total_detecciones = 0
    
    # B) Procesar cada detección con Fase 2 (Keras Multi-label)
    for i, box in enumerate(results[0].boxes):
        total_detecciones += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Recorte (Crop) con pequeño padding
        h, w = img.shape[:2]
        pad = 5
        cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
        cx2, cy2 = min(w, x2+pad), min(h, y2+pad)
        crop = img[cy1:cy2, cx1:cx2]
        
        if crop.size == 0: continue
            
        # Preparar para Keras
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_tf = tf.image.resize_with_pad(crop_rgb, IMG_SIZE, IMG_SIZE)
        crop_tf = tf.expand_dims(crop_tf, 0) 
        
        # Predicción de morfología (Multi-label)
        preds = clasificador.predict(crop_tf, verbose=0)[0]
        
        # Analizar predicciones
        defectos_detectados = []
        es_normal = False
        
        # Si el modelo es v8 (multi-label)
        for idx, prob in enumerate(preds):
            if prob > UMBRAL_MORFOLOGIA:
                stats[CLASES[idx]] += 1
                if CLASES[idx] == "normal":
                    es_normal = True
                else:
                    # Formatear nombre para el label visual
                    clean_name = CLASES[idx].split("_")[0].upper()
                    defectos_detectados.append(clean_name)
        
        # Elegir color y etiqueta visual
        if es_normal and not defectos_detectados:
            color = (0, 255, 0) # Verde
            label_txt = "NORMAL"
        else:
            color = (0, 100, 255) # Naranja/Rojo suave para defectos
            if defectos_detectados:
                label_txt = "|".join(defectos_detectados)
            else:
                label_txt = "ANORMAL"

        # C) Dibujar
        cv2.rectangle(img_anotada, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_anotada, f"{i}:{label_txt}", (x1, y1 - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- 📊 3. GUARDAR RESULTADO INDIVIDUAL Y CÁLCULO TZI ---
    total_anormales = total_detecciones - stats["normal"]
    total_defectos = stats["cabeza_anormal"] + stats["cola_anormal"] + stats["pieza_intermedia_anormal"] + stats["residuo_citoplasmatico"]
    tzi = (total_defectos / total_anormales) if total_anormales > 0 else 0
    
    # Dibujar panel de resumen
    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    overlay = img_anotada.copy()
    cv2.rectangle(overlay, (10, 10), (520, 250), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img_anotada, 0.3, 0, img_anotada)
    
    y0, dy = 40, 32
    cv2.putText(img_anotada, f"EXPERIMENTO 5 - TZI REPORT", (20, y0), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    
    y0 += dy
    cv2.putText(img_anotada, f"Muestra: {nombre_archivo[:30]}", (20, y0), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    y0 += dy
    cv2.putText(img_anotada, f"Células: {total_detecciones} | TZI: {tzi:.2f}", (20, y0), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    y0 += dy
    perc_normal = (stats["normal"] / total_detecciones * 100) if total_detecciones > 0 else 0
    cv2.putText(img_anotada, f"Normales: {stats['normal']} ({perc_normal:.1f}%)", (20, y0), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    y0 += dy
    # Mostrar desglose de defectos (como el Excel)
    cv2.putText(img_anotada, "Desglose de Defectos (sobre anormales):", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    col_x = [40, 260] # Dos columnas para ahorrar espacio
    current_col = 0
    
    for idx, clase in enumerate(CLASES):
        if clase == "normal": continue
        
        if current_col == 0: y0 += dy - 5
        
        count = stats[clase]
        perc = (count / total_anormales * 100) if total_anormales > 0 else 0
        name_short = clase.split("_")[0].capitalize()
        txt = f"- {name_short}: {perc:4.1f}%"
        
        cv2.putText(img_anotada, txt, (col_x[current_col], y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)
        current_col = 1 - current_col # Alternar columna

    ruta_salida = os.path.join(DIR_RESULTADOS, f"REPORTE_{nombre_archivo}")
    cv2.imwrite(ruta_salida, img_anotada)
    print(f"   ✅ Reporte detallado guardado en: {ruta_salida}")

print("\n✨ Proceso completo. Revisa la carpeta: docs/pruebas/resultados_v3")