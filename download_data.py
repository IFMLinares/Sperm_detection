import os
from dotenv import load_dotenv
from roboflow import Roboflow

# ==========================================
# ⚙️ CARGA DE CREDENCIALES DESDE ENTORNO
# ==========================================
# Prioridad: variables de entorno del sistema > archivo .env local
load_dotenv()  # carga desde .env si existe
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    print("\n❌ No se encontró la variable de entorno 'ROBOFLOW_API_KEY'.")
    print("   Soluciones:")
    print("   1) Crea un archivo .env con la línea: ROBOFLOW_API_KEY=TU_API_KEY")
    print("   2) O define la variable en Windows: setx ROBOFLOW_API_KEY TU_API_KEY")
    raise SystemExit(1)

print("⬇️ Conectando con Roboflow Universe...")

try:
    # 1. Autenticación
    rf = Roboflow(api_key=api_key)

    # 2. Selección del proyecto y versión
    print("   -> Buscando proyecto 'sperm-detection-mbcpn'...")
    project = rf.workspace("enes-tcjiz").project("sperm-detection-mbcpn")
    version = project.version(2)

    # 3. Descarga
    print("   -> Descargando dataset (esto puede tardar un poco)...")
    dataset = version.download("yolov8")

    print("\n✅ ¡ÉXITO! Dataset descargado.")
    print(f"   -> Las imágenes están en: {dataset.location}")
    print("   -> Ahora puedes ejecutar 'python train_hunter.py'")

except Exception as e:
    print(f"\n❌ ERROR DE DESCARGA: {e}")
    print("   (Verifica tu conexión a internet o si la API Key cambió)")