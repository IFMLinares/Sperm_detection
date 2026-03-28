# **🧬 Detección y Clasificación de Espermatozoides con IA (Fases 1 y 2)**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv11](https://img.shields.io/badge/Ultralytics-YOLOv11-3776AB)
![TensorFlow](https://img.shields.io/badge/TensorFlow%20Keras-EfficientNetB0-FF6F00)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Este proyecto implementa un sistema de Visión Artificial de grado clínico basado en los estándares del **Manual de la OMS (2021)** y morfología estricta de **Kruger**.

**Objetivo de la Fase 1:** Detección y localización automática en imágenes de microscopía (100x) utilizando **YOLOv11**.
**Objetivo de la Fase 2:** Clasificación de morfología multietiqueta (Normal + 4 tipos de anomalías) usando **EfficientNetB0 + Focal Loss**, validado con el índice **Cohen's Kappa**.

---

## **📋 Requisitos y Configuración**

1. **Entorno Conda:**
```bash
conda create -n tesis_espermas python=3.10 -y
conda activate tesis_espermas
pip install -r requirements.txt
```
2. **Hardware:** Recomendado NVIDIA GPU (RTX 3080/4090) con CUDA/cuDNN configurado.

---

## **🚀 Workflow del Proyecto**

### 1. Preparación de Datos
- **Descarga:** `python core/download_data.py`
- **Pre-procesamiento:** `python core/preparar_dataset_morfologia.py` (Genera los crops multietiqueta de 300px).

### 2. Entrenamiento (Estrategia v8 Ganadora)
- **Fase 1 (YOLO):** `python core/train_hunter.py`
- **Fase 2 (Morfología):** `python core/entrenar_morfologia.py`
  - Utiliza **EfficientNetB0**, **Focal Loss** y **Oversampling Físico** para manejar el desbalance de clases.
  - Los mejores pesos se guardan en: `models/trained/clasificacion/experimento_5/mejor_modelo_v8.h5`.

### 3. Validación y Métricas
- **Cálculo de Kappa:** `python core/calcular_kappa.py`
  - Protocolo de Tesis: Pool de validación (10%) -> Muestra aleatoria de 100 células.
  - Genera: Matriz de Confusión, Sensibilidad, Especificidad, Accuracy, VPP, VPN y F1-Score.

---

## **📂 Estructura del Proyecto Consolidado**

```
SpermDetection/  
│  
├── core/                   # Scripts principales de IA
│   ├── entrenar_morfologia.py    # Training oficial v8
│   ├── calcular_kappa.py        # Validación clínica oficial
│   └── analizador_completo.py    # Inferencia acoplada
│
├── models/
│   └── trained/
│       └── clasificacion/
│           └── experimento_5/     # Carpeta del MODELO GANADOR (v8)
│               └── mejor_modelo_v8.h5
│
├── docs/                   # Documentación y Resultdos
│   └── RESULTADOS_TESIS.md  # Reporte formal para sustentación
│
└── data/
    └── datasets/
        └── dataset_morfologia_v3/ # Dataset final multietiqueta
```

---
*Este repositorio ha sido consolidado para la entrega final de tesis, eliminando versiones experimentales obsoletas.*
