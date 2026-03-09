from ultralytics import YOLO
import os
import glob

# ==========================================
# CONFIGURACIÓN
# ==========================================
# Carpeta donde pondrás tus fotos de microscopio
CARPETA_ORIGEN = "../data/raw/my_images"
# Nombre del proyecto de salida
NOMBRE_SALIDA = "result_fase_1_1"
# Umbral de confianza (0.25 a 0.5 suele estar bien)
CONFIANZA = 0.25
# ==========================================

def main():
    # 1. Buscar el modelo entrenado automáticamente
    ruta_modelo = "runs/detect/trained_sperm_model/weights/best.pt"
    
    if not os.path.exists(ruta_modelo):
        print(f"ERROR: No encuentro el modelo en {ruta_modelo}")
        print("   -> ¿Terminó el entrenamiento? ¿El nombre de la carpeta en 'entrenar_cazador.py' es correcto?")
        return

    # 2. Verificar carpeta de imágenes
    if not os.path.exists(CARPETA_ORIGEN):
        os.makedirs(CARPETA_ORIGEN)
        print(f"La carpeta '{CARPETA_ORIGEN}' no existía. La he creado.")
        print(f"Por favor, mete tus fotos del microscopio dentro de '{CARPETA_ORIGEN}' y vuelve a ejecutar.")
        return

    archivos = glob.glob(os.path.join(CARPETA_ORIGEN, "*.*"))
    if not archivos:
        print(f"La carpeta '{CARPETA_ORIGEN}' está vacía.")
        return
    # 1. Buscar el modelo entrenado automáticamente
    ruta_modelo = "../models/trained/runs/detect/trained_sperm_model/weights/best.pt"
    
    if not os.path.exists(ruta_modelo):
        print(f"ERROR: No encuentro el modelo en {ruta_modelo}")
        print("   -> ¿Terminó el entrenamiento?")
        return

    print(f"🚀 Iniciando detección en {len(archivos)} imágenes...")
    print(f"   -> Modelo: {ruta_modelo}")

    # 3. Cargar modelo
    model = YOLO(ruta_modelo)

    # 4. Inferencia y Recorte
    # save=True: Dibuja los cuadros en la foto original
    # save_crop=True: Guarda cada esperma como una imagen individual (¡CRUCIAL!)
    results = model.predict(
        source=CARPETA_ORIGEN,
        conf=CONFIANZA,
        imgsz=1920, # <--- Obliga a YOLO a no achicar fotos de 4000x3000
        device=0,   # <--- Obliga a YOLO a usar la GPU 0 (la RTX 3080)
        iou=0.3,    # <--- Filtro para evitar detecciones dobles o fraccionadas
        save=True,
        save_crop=True, 
        project="../models/trained/runs/detect",
        name=NOMBRE_SALIDA,
        exist_ok=True
    )

    print("\n PROCESO TERMINADO!")
    output_path = f"../models/trained/runs/detect/{NOMBRE_SALIDA}"
    print(f"1. Fotos con cuadros dibujados: {output_path}/")
    print(f"2. ESPERMAS RECORTADOS:        {output_path}/crops/sperm/")
    print("\nVe a la carpeta 'crops' y revisa si recortó bien tus espermas.")

if __name__ == "__main__":
    main()