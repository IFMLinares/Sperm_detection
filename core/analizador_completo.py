import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import os
import glob

# --- ⚙️ CONFIGURACIÓN DE RUTAS ---
MODELO_YOLO = "../models/trained/runs/detect/trained_sperm_model/weights/best.pt"
MODELO_KERAS = "../models/trained/clasificacion/experimento_1/clasificador_morfologia_v1.keras"

# DIR_PRUEBAS = "../docs/pruebas/imagenes_microscopio"
DIR_PRUEBAS = "../data/raw/my_images"
DIR_RESULTADOS = "../docs/pruebas/resultados_20p"

IMG_SIZE = 224 # Tamaño para el clasificador Keras

# Crear carpeta de resultados si no existe
os.makedirs(DIR_RESULTADOS, exist_ok=True)

# --- 🧠 1. CARGAR INTELIGENCIA ---
print("🧠 Cargando modelos en la RTX 3080...")
# Configurar memoria GPU para evitar bloqueos
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

detector = YOLO(MODELO_YOLO)
clasificador = tf.keras.models.load_model(MODELO_KERAS)

# --- 🖼️ 2. PROCESAR TODAS LAS IMÁGENES ---
imagenes = glob.glob(os.path.join(DIR_PRUEBAS, "*.*"))

if not imagenes:
    print(f"⚠️ No se encontraron imágenes en {DIR_PRUEBAS}")
    exit()

print(f"🚀 Iniciando análisis de {len(imagenes)} muestras...")

for ruta_img in imagenes:
    nombre_archivo = os.path.basename(ruta_img)
    print(f"🔍 Analizando: {nombre_archivo}")
    
    img = cv2.imread(ruta_img)
    img_anotada = img.copy()
    
    # A) Detección Fase 1 (YOLO)
    results = detector.predict(img, conf=0.2, verbose=False)
    
    conteo = {"normal": 0, "anormal": 0}
    
    # B) Procesar cada detección con Fase 2 (Keras)
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Recorte (Crop)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue
            
        # Preparar para Keras (RGB + Padding + Expand Dims)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_tf = tf.image.resize_with_pad(crop_rgb, IMG_SIZE, IMG_SIZE)
        crop_tf = tf.expand_dims(crop_tf, 0) 
        
        # Predicción de morfología
        pred = clasificador.predict(crop_tf, verbose=0)[0][0]
        
        # Clasificar y Elegir Color (Verde=Normal, Rojo=Anormal)
        # Nota: Ajustamos según tus etiquetas (1 suele ser Normal)
        es_normal = pred > 0.5 
        label = "NORMAL" if es_normal else "ANORMAL"
        color = (0, 255, 0) if es_normal else (0, 0, 255)
        
        if es_normal: conteo["normal"] += 1
        else: conteo["anormal"] += 1

        # C) Dibujar recuadro e ID en la imagen completa
        cv2.rectangle(img_anotada, (x1, y1), (x2, y2), color, 3)
        texto = f"ID:{i} {label}"
        cv2.putText(img_anotada, texto, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # --- 📊 3. GUARDAR RESULTADO INDIVIDUAL ---
    total = conteo["normal"] + conteo["anormal"]
    porcentaje = (conteo["normal"] / total * 100) if total > 0 else 0
    
    # Dibujar cuadro de resumen en la esquina superior izquierda de la imagen
    resumen = f"Total: {total} | Normales: {conteo['normal']} ({porcentaje:.1f}%)"
    cv2.rectangle(img_anotada, (20, 20), (850, 80), (0, 0, 0), -1)
    cv2.putText(img_anotada, resumen, (40, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    ruta_salida = os.path.join(DIR_RESULTADOS, f"REPORTE_{nombre_archivo}")
    cv2.imwrite(ruta_salida, img_anotada)
    print(f"   ✅ Guardado: {ruta_salida} | Morfología: {porcentaje:.2f}%")

print("\n✨ ¡Proceso completo! Revisa la carpeta pruebas/resultados")