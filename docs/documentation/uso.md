# Guía de Uso Rápida (Windows)

## Preparación
- Instalación y entorno: ver [instalacion_windows.md](instalacion_windows.md).
- Instala las dependencias:
```
pip install -r requirements.txt
```
- Verifica la GPU (TensorFlow y PyTorch usan la RTX 3080):
```
python check_gpu.py
```
- Configura tu API Key de Roboflow en `.env` (ver [dataset.md](dataset.md)).

## Descarga de datos
```
python download_data.py
```
*(Nota: Para la Fase 2, asegúrate de tener poblada la carpeta `dataset_f2` manualmente).*

## Entrenamientos

**Fase 1 (YOLO - Detección):**
```bash
python core/train_hunter.py
```
Salida esperada: El mejor modelo se guarda automáticamente en `runs/detect/`.

**Fase 2 (Morfología - Clasificación):**
```bash
python core/entrenar_morfologia.py
```
Salida esperada: `models/trained/clasificacion/experimento_5/mejor_modelo_v8.h5`.

**Validación Técnica (Métricas de Tesis):**
```bash
python core/calcular_kappa.py
```
Genera un análisis estadístico de 100 muestras con el índice Kappa de Cohen.

## Análisis Full (Recomendado)
Para aplicar tanto YOLO (Detección) como EfficientNetB0 (Morfología) sobre tus imágenes:
```bash
python core/analizador_completo.py
```
- Imágenes procesadas con bounding boxes y resúmenes: `pruebas/resultados_20p/`

## Herramientas Base (Sólo Fase 1)

### Detección y recorte directo
```
python detect_crop.py
```
- Imágenes anotadas: `runs/detect/result_fase_1_1/`
- Recortes: `runs/detect/result_fase_1_1/crops/sperm/`

### Organización para re-anotación
```
python detect_organize.py
```
- Carpeta por imagen en `result_fase_1_20p/` (incluye mapa numerado y recortes).
- Dataset para Roboflow en `roboflow_dataset/`.

## Ajustes comunes
- `CONFIANZA` en `detect_crop.py` y `detect_organize.py`: 0.25–0.5 según calidad.
- `CARPETA_ORIGEN`: carpeta con tus fotos (por defecto `my_images`).
