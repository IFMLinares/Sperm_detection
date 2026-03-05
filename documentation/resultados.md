# Resultados y Salidas

## 1. Pipeline Analítico Integrado (`analizador_completo.py`)
La inferencia combinada de Detección (YOLO) y Morfología (Keras) se guarda en:
- `pruebas/resultados_20p/REPORTE_*.jpg`
- Estas imágenes contienen las detecciones recuadradas (colorizado por clase), etiquetas textuales (Normal/Anormal) y un letrero que sumariza el porcentaje/conteo final del campo semiótico particular.

## 2. Herramientas de Base / Re-etiquetado (YOLO solo)

### `detect_crop.py`
- Imágenes anotadas: `runs/detect/result_fase_1_1/`.
- Recortes por clase: `runs/detect/result_fase_1_1/crops/sperm/`.

### `detect_organize.py`
- Carpeta particular por imagen orginal en `result_fase_1_20p/`.
- Archivo conjunto `*_MAPA_NUMERADO.jpg` con rectángulos visuales ligados a los recortes extraídos en su propia carpeta.
- Imagenes y TXTs empacables a Roboflow se respaldan internamente en `roboflow_dataset/`.

## 3. Entrenamiento (Keras)
- Gráfico histórico sobre Accuracy y Loss (Evolución por Épocas) se empaqueta en `runs/clasificacion/experimento_1/reporte_entrenamiento.png`.

## Validación manual
- Revisa las clasificaciones visuales devueltas por el Analizador.
- Ajusta `CONFIANZA` en el generador YOLO o incrementa los Datasets de Fase 2 para pulir el dictamen en el caso de haber falsos positivos u oclusiones.
