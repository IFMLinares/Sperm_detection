# Entrenamiento de Modelos

Este proyecto entrena dos modelos independientes: un **detector de objetos (YOLO)** para ubicar espermatozoides (Fase 1) y un **clasificador de morfología (Keras/MobileNetV2)** para determinar si son normales o anormales (Fase 2).

## Fase 1: Entrenamiento del Detector (YOLOv8)
Este modelo permite identificar dónde hay un espermatozoide.

- Preferido: `yolov8s.pt` (buena precisión en objetos pequeños).
- Alternativa: `yolo11n.pt` (más rápido, menos preciso).

**Ejecución:**
```
python train_hunter.py
```
- Detecta `data.yaml` automáticamente.
- Usa GPU si está disponible (preferible); caso contrario, CPU con batch reducido.
- **Hiperparámetros**: `epochs=150`, `imgsz=640`, `patience=40` (early stopping). `batch=24` en GPU.
- **Salidas**: 
  - `runs/detect/trained_sperm_model/weights/best.pt`: mejor modelo.
  - `runs/detect/trained_sperm_model/weights/last.pt`: último checkpoint.

## Fase 2: Entrenamiento del Clasificador (Keras/MobileNetV2)
Este modelo analiza los recortes generados por la Fase 1 y clasifica y dictamina la morfología.

**Modelo base:** MobileNetV2 pre-entrenado en `ImageNet` (Transfer Learning), congelando sus capas base y añadiendo top layers especializadas (GlobalAveragePooling, Dropout(0.3) y Dense Sigmoid).

**Ejecución:**
```
python entrenar_modelo_f2.py
```
- Carga el dataset de recortes desde la carpeta `dataset_f2`.
- Realiza **Data Augmentation** dinámico (Flip, Rotación Aleatoria, Zoom Aleatorio) con capa Keras.
- Ajusta el redimensionado (`resize_with_pad` a 224x224) preservando proporciones originales.
- Maneja intrínsecamente el **Desequilibrio de Clases** dictando los pesos (p. ej., `class_weight={0: 3.6, 1: 1.0}`).
- **Hiperparámetros**: Óptimizador `Adam(1e-4)`, activador `Sigmoid` con pérdida `binary_crossentropy`, `batch_size=16`, `epochs=50`. Early Stopping en `patience=10`.
- **Salidas**: 
  - `runs/clasificacion/experimento_1/clasificador_morfologia_v1.keras`.
  - Reporte visual (`runs/clasificacion/experimento_1/reporte_entrenamiento.png`).

## Recomendaciones Generales
- "CUDA Out of Memory": reduce el `BATCH_SIZE` según el modelo empleado.
- La RTX 3080 ha sido configurada explícitamente (`memory_growth`) para que TensorFlow no monopolice el pool, garantizando tolerancia al usar el `analizador_completo.py` (cargando YOLO y Keras simultáneamente).
