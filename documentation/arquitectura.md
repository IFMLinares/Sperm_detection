# Arquitectura y Flujo (Fase 1)

Este proyecto implementa una canalización simple y reproducible para detección y recorte de espermatozoides.

## Componentes
- Dataset: descargado desde Roboflow en formato YOLOv8 (data.yaml, imágenes y labels).
- Entrenamiento: `train_hunter.py` con modelo base `yolov8s.pt` (opcional `yolo11n.pt`).
- Inferencia/Recorte: `detect_crop.py` genera anotaciones y crops.
- Organización: `detect_organize.py` crea carpetas por imagen y un mapa numerado.
- Utilidades: `check_gpu.py` para diagnóstico de entorno (Python, PyTorch, CUDA).

## Flujo de trabajo
1. Descarga dataset → `download_data.py`.
2. Entrena modelo → `train_hunter.py`.
3. Detecta y recorta → `detect_crop.py`.
4. Organiza resultados por imagen → `detect_organize.py`.

## Estructura de carpetas (simplificada)
```
SpermDetection/
├─ my_images/                 # imágenes de entrada (microscopio)
├─ runs/detect/
│  ├─ trained_sperm_model/    # pesos entrenados
│  └─ result_fase_1/          # inferencias y crops
├─ result_fase_1/             # organización por imagen + mapa numerado
└─ documentation/             # documentación del proyecto
```

## Notas
- Ajusta `CONFIANZA`, rutas y nombres en los scripts si tu estructura cambia.
- Mantén los pesos y datos bajo control de licencia cuando publiques en GitHub.
