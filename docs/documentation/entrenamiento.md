# Entrenamiento de Modelos

Este proyecto entrena dos modelos independientes: un **detector de objetos (YOLO)** para ubicar espermatozoides (Fase 1) y un **clasificador de morfología multietiqueta (Keras/EfficientNetB0)** para identificar anomalías específicas (Fase 2).

- Preferido: `yolo11n.pt` o `yolo11s.pt` (Arquitectura YOLOv11).

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

## Fase 2: Entrenamiento del Clasificador (EfficientNetB0)
Este modelo analiza los recortes de 300x300px y clasifica 5 categorías simultáneamente (Multietiqueta).

**Modelo base:** EfficientNetB0 pre-entrenado, utilizando una cabecera personalizada con `GlobalAveragePooling2D` y una capa `Dense` con activación `Sigmoid` para permitir múltiples etiquetas por célula.

**Ejecución:**
```bash
python core/entrenar_morfologia.py
```
- **Técnicas Avanzadas (v8):**
  - **Focal Loss:** Se utiliza `MultiLabelFocalLoss` para dar más peso a los ejemplos difíciles (anomalías raras) y reducir el impacto de ejemplos fáciles.
  - **Oversampling Físico:** El dataset se balancea antes de entrar a la red mediante la duplicación física de muestras de clases minoritarias.
  - **Resolución:** Se estandariza a **300x300px** para capturar detalles finos de la pieza intermedia y vacuolas.
- **Salidas**: 
  - `models/trained/clasificacion/experimento_5/mejor_modelo_v8.h5`.
  - Reporte de métricas y curvas de aprendizaje en la carpeta del experimento.

## Recomendaciones Generales
- "CUDA Out of Memory": reduce el `BATCH_SIZE` según el modelo empleado.
- La RTX 3080 ha sido configurada explícitamente (`memory_growth`) para que TensorFlow no monopolice el pool, garantizando tolerancia al usar el `analizador_completo.py` (cargando YOLO y Keras simultáneamente).
