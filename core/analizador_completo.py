import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import os
import glob
import json

# --- ⚙️ CONFIGURACIÓN DE RUTAS ---
# Usamos el modelo más reciente si existe, sino el anterior
MODELO_YOLO = "../models/trained/runs/detect/trained_sperm_model/weights/best.pt"
MODELO_KERAS = "../models/trained/clasificacion/experimento_3/clasificador_morfologia_v3.keras"
CLASES_JSON = "../models/trained/clasificacion/experimento_3/clases.json"

DIR_PRUEBAS = "../data/raw/my_images"
DIR_RESULTADOS = "../docs/pruebas/resultados_v3"

IMG_SIZE = 224 # Tamaño para el clasificador Keras
UMBRAL_MORFOLOGIA = 0.4 # Confianza mínima para etiquetar un defecto (0.1 a 0.9)
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
    try:
        clasificador = tf.keras.models.load_model(
            MODELO_KERAS,
            custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy}
        )
    except Exception:
        clasificador = tf.keras.models.load_model(MODELO_KERAS)
    # Cargar nombres de clases
    if os.path.exists(CLASES_JSON):
        with open(CLASES_JSON, "r", encoding="utf-8") as f:
            CLASES = json.load(f)
    else:
        CLASES = ["normal", "cabeza", "cola", "pieza_int", "residuo"]
else:
    print(f"⚠️ No se encontró el modelo Keras v3 en: {MODELO_KERAS}")
    print("Usando modo de compatibilidad con modelo v1 si existe...")
    MODELO_KERAS_V1 = "../models/trained/clasificacion/experimento_1/clasificador_morfologia_v1.keras"
    if os.path.exists(MODELO_KERAS_V1):
        clasificador = tf.keras.models.load_model(MODELO_KERAS_V1)
        CLASES = ["normal"] 
    else:
        print("❌ Ningún modelo de clasificación encontrado.")
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
        
        # Si el modelo es v3 (multi-label)
        if len(preds) > 1:
            # Usar el umbral personalizado para detectar defectos
            for idx, prob in enumerate(preds):
                if prob > UMBRAL_MORFOLOGIA:
                    stats[CLASES[idx]] += 1
                    if CLASES[idx] == "normal":
                        es_normal = True
                    else:
                        defectos_detectados.append(CLASES[idx].replace("_anormal", ""))
        else:
            # Compatibilidad con modelo v1 (binario)
            es_normal = preds[0] > 0.5
            if es_normal: stats["normal"] += 1
        
        # Elegir color y etiqueta visual
        if es_normal and not defectos_detectados:
            color = (0, 255, 0) # Verde
            label_txt = "NORMAL"
        else:
            color = (0, 0, 255) # Rojo
            if defectos_detectados:
                label_txt = ",".join(defectos_detectados).upper()
            else:
                label_txt = "ANORMAL"

        # C) Dibujar
        cv2.rectangle(img_anotada, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_anotada, f"{i}:{label_txt}", (x1, y1 - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- 📊 3. GUARDAR RESULTADO INDIVIDUAL ---
    # Dibujar panel de resumen
    overlay = img_anotada.copy()
    cv2.rectangle(overlay, (10, 10), (500, 210), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img_anotada, 0.4, 0, img_anotada)
    
    y0, dy = 40, 30
    cv2.putText(img_anotada, f"Muestra: {nombre_archivo}", (20, y0), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    y0 += dy
    perc_normal = (stats["normal"] / total_detecciones * 100) if total_detecciones > 0 else 0
    cv2.putText(img_anotada, f"Total: {total_detecciones} | Normales: {perc_normal:.1f}%", (20, y0), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    y0 += dy
    # Mostrar top defectos
    defectos_str = "Defectos:"
    cv2.putText(img_anotada, defectos_str, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    for idx, clase in enumerate(CLASES):
        if clase == "normal": continue
        y0 += dy - 5
        count = stats[clase]
        perc = (count / total_detecciones * 100) if total_detecciones > 0 else 0
        txt = f"- {clase.replace('_anormal', ''):15s}: {count:3d} ({perc:4.1f}%)"
        cv2.putText(img_anotada, txt, (40, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 1)

    ruta_salida = os.path.join(DIR_RESULTADOS, f"DETALLE_{nombre_archivo}")
    cv2.imwrite(ruta_salida, img_anotada)
    print(f"   ✅ Reporte detallado guardado en: {ruta_salida}")

print("\n✨ Proceso completo. Revisa la carpeta: docs/pruebas/resultados_v3")