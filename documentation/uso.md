# Guía de Uso Rápida (Windows)

## Preparación
- Instalación y entorno: ver [instalacion_windows.md](instalacion_windows.md).
- Instala dependencias:
```
pip install -r requirements.txt
```
- Verifica GPU y entorno:
```
python check_gpu.py
```
- Configura tu API Key de Roboflow en `.env` (ver [dataset.md](dataset.md)).

## Descarga del dataset
```
python download_data.py
```

## Entrenamiento
```
python train_hunter.py
```
Salida esperada: `runs/detect/trained_sperm_model/weights/best.pt`.

## Detección y recorte
```
python detect_crop.py
```
- Imágenes anotadas: `runs/detect/result_fase_1/`
- Recortes: `runs/detect/result_fase_1/crops/sperm/`

## Organización por imagen
```
python detect_organize.py
```
- Carpeta por imagen en `result_fase_1/`
- Mapa numerado: `*_MAPA_NUMERADO.jpg`

## Ajustes comunes
- `CONFIANZA` (umbral): 0.25–0.5 según calidad.
- `CARPETA_ORIGEN`: carpeta con tus fotos (por defecto `my_images`).
