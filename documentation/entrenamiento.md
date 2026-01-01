# Entrenamiento del Modelo

## Modelo base
- Preferido: `yolov8s.pt` (buena precisión en objetos pequeños).
- Alternativa: `yolo11n.pt` (más rápido, menos preciso).

## Ejecución
```
python train_hunter.py
```
- Detecta `data.yaml` automáticamente.
- Usa GPU si está disponible; caso contrario, CPU con batch reducido.

## Hiperparámetros (por defecto en `train_hunter.py`)
- `epochs=150`, `imgsz=640`, `patience=40` (early stopping).
- `batch=24` en GPU; `batch` menor en CPU.
- `augment=True` para mejorar generalización en datasets pequeños.

## Salidas
- `runs/detect/trained_sperm_model/weights/best.pt`: mejor modelo.
- `runs/detect/trained_sperm_model/weights/last.pt`: último checkpoint.

## Recomendaciones
- Si hay "CUDA Out of Memory": baja `batch`.
- Mantén el dataset balanceado y con anotaciones consistentes.
