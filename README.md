# **🧬 Detección de Espermatozoides con IA (Fase 1: Detección)**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-3776AB)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Este proyecto implementa un sistema de Visión Artificial basado en los estándares del **Manual de la OMS (2021)** para el análisis de semen.

**Objetivo de la Fase 1:** Detectar y recortar automáticamente espermatozoides individuales en imágenes de microscopía (100x) utilizando **YOLOv8** y aceleración por GPU (USADA PARA EL ENTRENAMIENTO: NVIDIA RTX 3080).

## **📋 Requisitos y Configuración**

Resumen rápido; guía completa en [documentation/instalacion_windows.md](documentation/instalacion_windows.md).
- Miniconda o Anaconda, VS Code, drivers NVIDIA (si usarás GPU).
- Crea el entorno y instala dependencias:
```
conda create -n tesis_espermas python=3.10 -y
conda activate tesis_espermas
pip install -r requirements.txt
```

Para problemas de PATH, permisos y ToS, ver [documentation/instalacion_windows.md](documentation/instalacion_windows.md) y [documentation/troubleshooting.md](documentation/troubleshooting.md).

## **🚀 Quickstart**
1) Instala dependencias: `pip install -r requirements.txt`
2) Configura tu API Key en `.env` (ver [documentation/dataset.md](documentation/dataset.md)).
3) Descarga dataset: `python download_data.py`
4) Entrena: `python train_hunter.py`
5) Detecta y recorta: `python detect_crop.py`
6) Organiza resultados: `python detect_organize.py`

## **📚 Documentación**
- [Guía de Uso](documentation/uso.md)
- [Instalación en Windows](documentation/instalacion_windows.md)
- [Dataset y `.env`](documentation/dataset.md)
- [Entrenamiento](documentation/entrenamiento.md)
- [Resultados](documentation/resultados.md)
- [Arquitectura](documentation/arquitectura.md)
- [FAQ](documentation/faq.md) · [Troubleshooting](documentation/troubleshooting.md)

## **🧠 Entrenamiento y Detección**
- Entrenamiento: ver [documentation/entrenamiento.md](documentation/entrenamiento.md).
- Detección y organización: ver [documentation/uso.md](documentation/uso.md) y [documentation/resultados.md](documentation/resultados.md).

---

## **📤 Releases del modelo**
Publica `best.pt` como asset de Release (no dentro del repo). Ver pasos en [sección de publicación](#-10-publicar-el-modelo-como-release-github).

---


## **🗺️ Roadmap Fase 2 (Clasificación/Morfología)**

- Conteo robusto y métricas por campo (confiables para reporte).
- Clasificación de morfología normal/anormal y motilidad.
- Post-procesamiento y exportación (CSV/JSON) por muestra.
- Panel simple de visualización y QA de anotaciones.

---
Para errores y soluciones, consulta [documentation/troubleshooting.md](documentation/troubleshooting.md).

## **📂 Estructura del Proyecto**

Una vez ejecutados los scripts, tu carpeta debería verse así:

DetectorEspermas/  
│  
├── README.md            \# Este archivo de documentación  
├── descargar\_data.py    \# Script de descarga  
├── entrenar.py          \# Script de entrenamiento  
├── Sperm-detection.../  \# Carpeta generada por Roboflow (puede variar nombre)  
│   ├── train/           \# Imágenes de entrenamiento  
│   ├── valid/           \# Imágenes de validación  
│   └── data.yaml        \# Configuración del dataset  
│  
└── runs/                \# RESULTADOS DEL ENTRENAMIENTO  
    └── detect/  
        └── cazador\_espermas\_v1/  
            └── weights/  
                ├── best.pt  \<-- TU MODELO FINAL (USAR ESTE PARA DETECTAR)  
                └── last.pt  
