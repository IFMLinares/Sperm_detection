# Arquitectura y Flujo (Fases 1 y 2)

Este proyecto implementa una canalización reproducible para detección, recorte y clasificación morfológica de espermatozoides.

## Componentes
- **Dataset (Detección)**: Descargado desde Roboflow (YOLOv8).
- **Dataset (Clasificación)**: Carpeta local `dataset_f2/` dividida en clases (Normal/Anormal).
- **Fase 1 (Entrenamiento)**: `train_hunter.py` con modelo base `yolo11n.pt` (Arquitectura YOLOv11).
- **Fase 2 (Entrenamiento)**: `entrenar_morfologia.py` (Keras/EfficientNetB0 con Data Augmentation, Focal Loss y Oversampling Físico).
- **Validación Clínica**: `calcular_kappa.py` calcula métricas detalladas (Kappa, Sensibilidad, Especificidad, VPN) sobre una muestra de 100 células.
- **Inferencia Integrada (Analizador)**: `analizador_completo.py` acopla YOLOv11 y EfficientNetB0 para evaluación morfológica automática.
- **Diagnóstico**: `check_gpu.py` para diagnóstico de entorno (Python, PyTorch, TensorFlow).

## Flujo de trabajo Completo
1. Descarga el dataset base → `download_data.py`.
2. Entrena modelo de Detección (YOLO) → `train_hunter.py`.
3. Entrena modelo de Clasificación (Keras) → `entrenar_morfologia.py`.
4. Valida resultados clínicos → `calcular_kappa.py`.
5. Evalúa imágenes reales con inferencia dual → `analizador_completo.py`.

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
