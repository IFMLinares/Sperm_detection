# **🧬 Detección de Espermatozoides con IA (Fases 1 y 2)**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-3776AB)
![TensorFlow](https://img.shields.io/badge/TensorFlow%20Keras-MobileNetV2-FF6F00)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Este proyecto implementa un sistema de Visión Artificial basado en los estándares del **Manual de la OMS (2021)** para el análisis de semen.

**Objetivo de la Fase 1:** Detectar y recortar automáticamente espermatozoides individuales en imágenes de microscopía (100x) utilizando **YOLOv8**.
**Objetivo de la Fase 2:** Clasificar la morfología de los espermatozoides detectados (Normal/Anormal) usando **Keras y MobileNetV2**, generando un reporte visual integrado por muestra.

Ambas fases aprovechan aceleración por GPU (NVIDIA RTX 3080).

## **📋 Requisitos y Configuración**

Resumen rápido; guía completa en [documentation/instalacion_windows.md](documentation/instalacion_windows.md).
- Miniconda o Anaconda, VS Code, drivers NVIDIA (si usarás GPU).
- Crea el entorno e instala dependencias:
```
conda create -n tesis_espermas python=3.10 -y
conda activate tesis_espermas
pip install -r requirements.txt
```

Para problemas de PATH, permisos y ToS, ver [documentation/instalacion_windows.md](documentation/instalacion_windows.md) y [documentation/troubleshooting.md](documentation/troubleshooting.md).

## **🚀 Quickstart**

### Preparación de Datos
1) Instala dependencias: `pip install -r requirements.txt`
2) Configura tu API Key en `.env` para la Fase 1 (ver [documentation/dataset.md](documentation/dataset.md)).
3) Descarga dataset de Fase 1: `python download_data.py`. Asegúrate de tener listo el `dataset_f2` para la Fase 2.

### Entrenamiento
4) Entrena el Detector (Fase 1 - YOLO): `python train_hunter.py`
5) Entrena el Clasificador (Fase 2 - Keras): `python entrenar_modelo_f2.py`

### Ejecución
6) Análisis Completo (Fase 1 + Fase 2): `python analizador_completo.py`
*Alternativamente, usando solo componentes de la Fase 1:*
- Detectar y recortar (YOLO): `python detect_crop.py`
- Organizar para anotación asistida (YOLO): `python detect_organize.py`

## **📚 Documentación**
- [Guía de Uso](documentation/uso.md)
- [Instalación en Windows](documentation/instalacion_windows.md)
- [Datasets (`.env` y Fase 2)](documentation/dataset.md)
- [Entrenamiento (YOLO y Keras)](documentation/entrenamiento.md)
- [Resultados](documentation/resultados.md)
- [Arquitectura](documentation/arquitectura.md)
- [FAQ](documentation/faq.md) · [Troubleshooting](documentation/troubleshooting.md)

## **🧠 Entrenamiento y Detección**
- Detalles de entrenamiento (YOLO y MobileNetV2): ver [documentation/entrenamiento.md](documentation/entrenamiento.md).
- Tiempos de inferencia, inferencia acoplada y organización: ver [documentation/uso.md](documentation/uso.md) y [documentation/resultados.md](documentation/resultados.md).

---
Para errores y soluciones, consulta [documentation/troubleshooting.md](documentation/troubleshooting.md).

## **📂 Estructura del Proyecto**

Una vez ejecutados los scripts y generados los reportes, tu carpeta principal debería verse así:

```
DetectorEspermas/  
│  
├── README.md               # Este archivo de documentación  
├── download_data.py        # Descarga dataset Fase 1
├── train_hunter.py         # Entrena YOLOv8 (Fase 1)
├── entrenar_modelo_f2.py   # Entrena Keras/MobileNetV2 (Fase 2)
├── detect_crop.py          # Detección y recortes YOLO (Base)
├── detect_organize.py      # Ordena resultados YOLO
├── analizador_completo.py  # Acopla YOLO y Keras para análisis full
│
├── my_images/              # Tus imágenes de microscopio a evaluar 
├── dataset_f2/             # Dataset clasificado Normal/Anormal (Fase 2)
│
├── runs/                   # RESULTADOS DE ENTRENAMIENTO
│   ├── detect/             # Modelos YOLO
│   └── clasificacion/      # Modelos Keras
│  
└── pruebas/  
    └── resultados_20p/     # Output visual del analizador_completo.py
```
