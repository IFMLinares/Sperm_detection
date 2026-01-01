from ultralytics import YOLO
import os
import sys
import torch

def encontrar_yaml():
    """Busca automáticamente el archivo data.yaml en subcarpetas."""
    print("🔍 Buscando archivo de configuración del dataset (data.yaml)...")
    for root, dirs, files in os.walk("."):
        if "data.yaml" in files:
            ruta_completa = os.path.join(root, "data.yaml")
            print(f"   -> Encontrado: {ruta_completa}")
            return ruta_completa
    return None

def main():
    # 1. Verificar Dataset
    yaml_path = encontrar_yaml()
    
    if yaml_path is None:
        print("\n❌ ERROR: No se encontró 'data.yaml'.")
        print("   -> ¿Ya ejecutaste 'python descargar_data.py'?")
        sys.exit(1)

    # 2. Cargar Modelo Base
    # Usamos YOLOv8 Small (s) para mejor detección de objetos pequeños como espermas
    print("\n📦 Cargando modelo YOLOv8 Small...")
    model = YOLO('yolov8s.pt')

    # 3. Selección de dispositivo (GPU si disponible, si no CPU)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device_arg = 0  # primera GPU
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n🔥 INICIANDO ENTRENAMIENTO EN GPU ({gpu_name})...")
    else:
        device_arg = 'cpu'
        print("\n🔥 INICIANDO ENTRENAMIENTO EN CPU...")
    print("   (Esto puede tomar unos minutos. Presiona Ctrl+C para cancelar si es necesario)\n")
    
    try:
        results = model.train(
            data=yaml_path,
            epochs=150,      # Suficiente para datasets < 1000 imágenes
            imgsz=640,       # Resolución estándar
            device=device_arg,  # GPU si disponible, de lo contrario CPU
            
            # --- Optimización de Hardware ---
            batch=24 if use_cuda else 8,  # reducir batch si se entrena en CPU
            workers=8,       # Tu Ryzen tiene 6 núcleos físicos, usamos 8 hilos para cargar datos rápido
            
            # --- Ajustes de Entrenamiento ---
            patience=40,     # Early Stopping: Si no mejora en 40 épocas, para.
            project='runs/detect',
            name='trained_sperm_model',
            exist_ok=True,   # Sobrescribe si ya existe la carpeta para no llenar disco
            augment=True,    # Vital para pocos datos: crea variaciones artificiales
            
            # --- Visualización ---
            verbose=True
        )
        
        print("\n🏁 ¡ENTRENAMIENTO FINALIZADO!")
        print(f"   -> Tu mejor modelo está en: runs/detect/trained_sperm_model/weights/best.pt")
        print("   -> Úsalo ahora para recortar tus propias imágenes.")

    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        print(f"\nDetalles del entorno:")
        print(f"torch.__version__: {torch.__version__}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"os.environ.get('CUDA_VISIBLE_DEVICES'): {os.environ.get('CUDA_VISIBLE_DEVICES')}")

if __name__ == '__main__':
    # Esta guarda es necesaria en Windows para evitar bucles infinitos con multiprocessing
    main()