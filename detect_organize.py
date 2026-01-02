from ultralytics import YOLO
import cv2
import os
import glob
from pathlib import Path

# ==========================================
# ⚙️ CONFIGURACIÓN
# ==========================================
CARPETA_ORIGEN = "my_images"       # Donde pones tus fotos
CARPETA_SALIDA = "result_fase_1"   # Donde saldrá todo ordenado
CONFIANZA = 0.4                    # Nivel de exigencia
# ==========================================

def main():
    # 1. Cargar modelo
    ruta_modelo = "runs/detect/trained_sperm_model/weights/best.pt"
    
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

    # 3. Procesar una por una
    for img_path in imagenes:
        nombre_base = Path(img_path).stem
        
        # Crear carpeta específica
        ruta_carpeta_img = os.path.join(CARPETA_SALIDA, nombre_base)
        os.makedirs(ruta_carpeta_img, exist_ok=True)
        
        print(f"   -> Analizando: {nombre_base}...")

        # Inferencia
        results = model.predict(img_path, conf=CONFIANZA, verbose=False)
        result = results[0]

        # Cargamos la imagen original dos veces:
        # 1. img_original: Limpia, para recortar los espermas sin rayas pintadas.
        # 2. img_anotada: Para dibujar los cuadrados y los números.
        img_original = cv2.imread(img_path)
        img_anotada = img_original.copy()
        
        cont = 0
        if result.boxes:
            for box in result.boxes:
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
        
        print(f"      ✅ Encontrados: {cont} espermas. Mapa guardado.")

    print("\n🏁 ¡PROCESO TERMINADO!")
    print(f"📂 Revisa la carpeta: {CARPETA_SALIDA}")

if __name__ == "__main__":
    main()