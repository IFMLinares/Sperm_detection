# Datasets y Anotaciones

Este proyecto requiere dos orígenes de datos, uno para cada fase.

## Fase 1: Dataset de Detección (Roboflow + YOLOv8)

### Fuente
- Roboflow Universe: proyecto `sperm-detection-mbcpn` versión 2 (formato YOLOv8).
- Descarga: ejecutando `python download_data.py`.

### Variables de entorno / `.env`
- Define tu API Key de Roboflow en el archivo `.env` (ver [../.env.example](../.env.example)):
```env
ROBOFLOW_API_KEY=TU_API_KEY
```
- Alternativa persistente en Windows (CMD):
```cmd
setx ROBOFLOW_API_KEY TU_API_KEY
```

### Estructura YOLOv8
- Contiene carpetas `train/`, `valid/`, (opcional `test/`).
- Incluye el archivo `data.yaml` especificando rutas y mapeos al motor Ultralytics.
- **Clase:** `sperm`.

## Fase 2: Dataset de Clasificación (Keras)

### Estructura `dataset_f2`
Para entrenar la red dictaminadora (Phase 2), se usa un enfoque clásico de carpetas bajo `dataset_f2/`:
```text
dataset_f2/
├── normal/         # Recortes (.jpg/.png) de espermatozoides sanos
└── anormal/        # Recortes con anomalías de cabeza/cuello/cola
```

Este directorio es alimentado manual o indirectamente a partir de selecciones post-procesadas generadas en la Fase 1 (`crops/sperm` producidos por `detect_crop.py` o `detect_organize.py`). Keras usará `image_dataset_from_directory` para crear inferencias de entrenamiento y validación al invocar `entrenar_modelo_f2.py`.
