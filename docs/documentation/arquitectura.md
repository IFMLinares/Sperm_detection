# Arquitectura y Flujo (Fases 1 y 2)

Este proyecto implementa una canalización reproducible para detección, recorte y clasificación morfológica de espermatozoides.

## Componentes
- **Dataset (Detección)**: Descargado desde Roboflow (YOLOv8).
- **Dataset (Clasificación)**: Carpeta local `dataset_f2/` dividida en clases (Normal/Anormal).
- **Fase 1 (Entrenamiento)**: `train_hunter.py` con modelo base `yolov8s.pt`.
- **Fase 2 (Entrenamiento)**: `entrenar_modelo_f2.py` (Keras/MobileNetV2 con Data Augmentation, ponderación de clases y Early Stopping).
- **Inferencia Separada (Fase 1)**: `detect_crop.py` genera anotaciones y crops.
- **Organización (Fase 1)**: `detect_organize.py` crea carpetas por imagen, un mapa numerado y copias para Roboflow.
- **Inferencia Integrada (Analizador)**: `analizador_completo.py` acopla YOLOv8 (encuentra objetos) y MobileNetV2 (evalúa dictamen morfológico en cada recorte).
- **Diagnóstico**: `check_gpu.py` para diagnóstico de entorno (Python, PyTorch, TensorFlow).

## Flujo de trabajo Completo
1. Descarga el dataset base → `download_data.py`.
2. Entrena modelo de Detección (YOLO) → `train_hunter.py`.
3. Entrena modelo de Clasificación (Keras) → `entrenar_modelo_f2.py`.
4. Evalúa imágenes reales con inferencia dual → `analizador_completo.py`.

*(Flujo alterno para re-etiquetar)*: `detect_crop.py` o `detect_organize.py` → Subida manual a Roboflow/Dataset_F2.

## Estructura de carpetas (Global)
```text
SpermDetection/
├─ my_images/                 # imágenes de entrada (microscopio)
├─ dataset_f2/                # datos para entrenar clasificador
├─ runs/
│  ├─ detect/                 # modelos y anotaciones YOLO
│  └─ clasificacion/          # modelos y gráficas Keras
├─ pruebas/
│  └─ resultados_20p/         # output analítico con YOLO+Keras
├─ result_fase_1_1/           ... (herramientas legacy/específicas)
├─ result_fase_1_20p/         ...
└─ documentation/             # documentación del proyecto
```

## Notas
- Ambas redes (YOLO y Keras) coexisten en memoria GPU durante el pipeline completo (`analizador_completo.py`). TensorFlow está configurado explícitamente con `memory_growth` para evitar colisiones por saturación de VRAM (en la RTX 3080).
- Mantén los pesos (`.pt`, `.keras`) y datos bajo control de licencia cuando publiques en repositorios remotos.
