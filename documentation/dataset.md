# Dataset y Anotación

## Fuente
- Roboflow Universe: proyecto `sperm-detection-mbcpn` versión 2 (formato YOLOv8).
- Descarga: `python download_data.py`.

## Variables de entorno / `.env`
- Define tu API Key en `.env` (ver [../.env.example](../.env.example)):
```
ROBOFLOW_API_KEY=TU_API_KEY
```
- Alternativa persistente en Windows:
```
setx ROBOFLOW_API_KEY TU_API_KEY
```
El script `download_data.py` leerá `ROBOFLOW_API_KEY` desde el entorno o `.env`.

## Estructura YOLOv8
- Carpetas `train/`, `valid/`, (opcional `test/`).
- Archivo `data.yaml` con rutas y clases.

## Clases
- `sperm` (actual).
- Fase 2 podría añadir clases de morfología: normal/anormal, motilidad.
