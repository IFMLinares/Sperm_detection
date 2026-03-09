from ultralytics import YOLO
import cv2
import os
import glob
import shutil
from pathlib import Path

# ==========================================
# ⚙️ CONFIGURACIÓN
# ==========================================
CARPETA_ORIGEN = "../data/raw/my_images"       # Donde pones tus fotos
CARPETA_SALIDA = "../data/processed/result_fase_1_20p"   # Donde saldrá todo ordenado
CARPETA_ROBOFLOW = "../data/datasets/roboflow_dataset"  # Carpeta lista para arrastrar a Roboflow
CONFIANZA = 0.25                    # Nivel de exigencia
# ==========================================

def main():
    # 1. Cargar modelo
    ruta_modelo = "../models/trained/runs/detect/trained_sperm_model/weights/best.pt"
    
    if not os.path.exists(ruta_modelo):
        print("❌ No encuentro el modelo 'best.pt'.")
        return

    print(f"🧠 Cargando modelo: {ruta_modelo}")
    model = YOLO(ruta_modelo)

    # 2. Buscar imágenes
    imagenes = glob.glob(os.path.join(CARPETA_ORIGEN, "*.*"))
    extensiones_validas = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    imagenes = [f for f in imagenes if Path(f).suffix.lower() in extensiones_validas]

    if not imagenes:
        print(f"⚠️ No hay imágenes en '{CARPETA_ORIGEN}'.")
        return

    print(f"🚀 Procesando {len(imagenes)} imágenes...")

    # Crear la carpeta de exportación a Roboflow
    os.makedirs(CARPETA_ROBOFLOW, exist_ok=True)

    # 3. Procesar una por una
    for img_path in imagenes:
        nombre_base = Path(img_path).stem
        
        # Validar que la imagen no esté corrupta
        img_original = cv2.imread(img_path)
        if img_original is None:
            print(f"   ⚠️ ERROR: No se puede leer la imagen {nombre_base} (posible archivo corrupto). Saltando...")
            continue
            
        # Crear carpeta específica
        ruta_carpeta_img = os.path.join(CARPETA_SALIDA, nombre_base)
        os.makedirs(ruta_carpeta_img, exist_ok=True)
        
        print(f"   -> Analizando: {nombre_base}...")

        # Inferencia
        results = model.predict(img_path, conf=CONFIANZA, imgsz=1920, device=0, iou=0.3, verbose=False)
        result = results[0]

        # === EXPORTACIÓN PARA ROBOFLOW ===
        ruta_txt_roboflow = os.path.join(CARPETA_ROBOFLOW, f"{nombre_base}.txt")
        ruta_img_roboflow = os.path.join(CARPETA_ROBOFLOW, Path(img_path).name)
        
        # Copiamos la imagen limpia a la carpeta conjunta
        shutil.copy(img_path, ruta_img_roboflow)
        
        # Abrimos el txt para guardar las coordenadas
        txt_file = open(ruta_txt_roboflow, "w")
        # =================================

        # Creamos una copia para dibujar la imagen mapa
        img_anotada = img_original.copy()
        
        cont = 0
        if result.boxes:
            for box in result.boxes:
                # --- EXPORTAR A TXT (Formato YOLO Normalizado) ---
                xn, yn, wn, hn = box.xywhn[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                txt_file.write(f"{cls_id} {xn:.6f} {yn:.6f} {wn:.6f} {hn:.6f}\n")
                
                # Coordenadas
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # --- A) RECORTAR (De la imagen limpia) ---
                crop = img_original[y1:y2, x1:x2]
                
                if crop.size > 0:
                    # Guardamos el recorte con el número ID
                    nombre_recorte = f"{nombre_base}_esperma_{cont}.jpg"
                    ruta_recorte = os.path.join(ruta_carpeta_img, nombre_recorte)
                    cv2.imwrite(ruta_recorte, crop)

                    # --- B) DIBUJAR EN LA IMAGEN GENERAL ---
                    # 1. Dibujar rectángulo verde neón
                    color_box = (0, 255, 0) # BGR
                    cv2.rectangle(img_anotada, (x1, y1), (x2, y2), color_box, 2)

                    # 2. Dibujar etiqueta con el número
                    label = f"ID: {cont}"
                    
                    # Fondo negro pequeño para que se lea el número
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(img_anotada, (x1, y1 - 20), (x1 + w, y1), color_box, -1)
                    
                    # Texto negro sobre el fondo verde
                    cv2.putText(img_anotada, label, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                    cont += 1
        
        # Guardar la imagen mapa completa
        ruta_mapa = os.path.join(ruta_carpeta_img, f"{nombre_base}_MAPA_NUMERADO.jpg")
        cv2.imwrite(ruta_mapa, img_anotada)
        
        txt_file.close()
        
        print(f"      ✅ Encontrados: {cont} espermas. Mapa y TXT guardados.")

    print("\n🏁 ¡PROCESO TERMINADO!")
    print(f"📂 Recortes y mapas en: {CARPETA_SALIDA}")
    print(f"🚀 LISTO PARA SUBIR A ROBOFLOW: {CARPETA_ROBOFLOW}")

if __name__ == "__main__":
    main()