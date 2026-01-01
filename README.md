# **🧬 Detección de Espermatozoides con IA (Fase 1: Detección)**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-3776AB)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Este proyecto implementa un sistema de Visión Artificial basado en los estándares del **Manual de la OMS (2021)** para el análisis de semen.

**Objetivo de la Fase 1:** Detectar y recortar automáticamente espermatozoides individuales en imágenes de microscopía (100x) utilizando **YOLOv8** y aceleración por GPU (USADA PARA EL ENTRENAMIENTO: NVIDIA RTX 3080).

## **📋 1\. Requisitos Previos**

Antes de comenzar, asegúrate de tener instalado:

1. **Miniconda O Anaconda** (Gestor de entornos Python, En este caso se utilizó Miniconda).  
2. **Visual Studio Code** (Editor de código).  
3. **Drivers de NVIDIA** actualizados (para soporte CUDA).  
4. **Hardware Recomendado:** GPU NVIDIA con al menos 4GB de VRAM (Probado en RTX 3080 10GB).

## **⚙️ 2\. Configuración Inicial del Sistema (Solo primera vez)**

Si es la primera vez que usas Conda en Windows con VS Code, debes realizar estos ajustes críticos para evitar errores de permisos, rutas y licencias.

### **A. Agregar Conda al PATH de Windows**

Si la terminal no reconoce el comando conda, debes agregarlo manualmente a las variables de entorno (Se usará de ejemplo la ruta de miniconda):

1. Busca en Windows **"Editar las variables de entorno de esta cuenta"**.  
2. Edita la variable Path y agrega estas 3 rutas (cambia TU\_USUARIO por tu nombre de usuario real):  
   * C:\\Users\\TU\_USUARIO\\miniconda3  
   * C:\\Users\\TU\_USUARIO\\miniconda3\\Scripts  
   * C:\\Users\\TU\_USUARIO\\miniconda3\\Library\\bin  
3. **Reinicia** Visual Studio Code para aplicar los cambios.

### **B. Inicializar PowerShell y Permisos**

Abre una terminal en VS Code (Ctrl \+ ñ) y ejecuta estos comandos uno por uno para permitir la ejecución de scripts y vincular Conda:

\# 1\. Permitir scripts (Soluciona error rojo de seguridad "UnauthorizedAccess")  
Set-ExecutionPolicy RemoteSigned \-Scope CurrentUser

\# 2\. Inicializar Conda en PowerShell  
conda init powershell

*⚠️ **IMPORTANTE:** Cierra la terminal actual y abre una nueva para ver reflejados los cambios.*

### **C. Aceptar Términos de Servicio (Solución a error 'CondaToSNonInteractiveError')**

Anaconda requiere aceptar explícitamente las licencias de sus canales. Ejecuta estos comandos si recibes errores al crear el entorno:

conda tos accept \--override-channels \--channel \[https://repo.anaconda.com/pkgs/main\](https://repo.anaconda.com/pkgs/main)  
conda tos accept \--override-channels \--channel \[https://repo.anaconda.com/pkgs/r\](https://repo.anaconda.com/pkgs/r)  
conda tos accept \--override-channels \--channel \[https://repo.anaconda.com/pkgs/msys2\](https://repo.anaconda.com/pkgs/msys2)

## **🚀 3\. Creación del Entorno Virtual**

Para generar un entorno aislado y cargar las librerías de IA

\# 1\. Crear el entorno (Python 3.10 es la versión más estable para YOLOv8)  
conda create \-n tesis\_espermas python=3.10 \-y

\# 2\. Activar el entorno  
conda activate tesis\_espermas  
\# (IMPORTANTE: Debes ver '(tesis\_espermas)' en verde a la izquierda de tu terminal)

\# 3\. Instalar librerías principales  
pip install ultralytics roboflow

**Recomendado Instalar requirements.txt para mejor compatibiliodad con el proyecto:**

```
 pip install -r requirements.txt
```

## **📥 4\. Descarga del Dataset (Script existente)**

Usa el script ya incluido [download_data.py](download_data.py) para descargar automáticamente el dataset etiquetado desde Roboflow Universe y organizarlo en formato YOLOv8.

- El script requiere tu API Key privada de Roboflow.
- Registra tu APIKEY de Roboflow en el archivo .env

**Ejecutar descarga:**

```
python download_data.py
```

## **🧠 5\. Entrenamiento del Modelo (Script existente)**

Usa el script [train_hunter.py](train_hunter.py) para el Fine-Tuning con **Ultralytics YOLOv8**. El script detecta automáticamente `data.yaml`, elige GPU si está disponible y guarda el mejor modelo en `runs/detect/trained_sperm_model/weights/best.pt`.

**Iniciar entrenamiento:**

```
python train_hunter.py
```

**Modelos base disponibles:**
- [yolov8s.pt](yolov8s.pt): equilibrado para objetos pequeños (recomendado).
- [yolo11n.pt](yolo11n.pt): versión ligera de YOLOv11 (rápida, menos precisa).

Si tu GPU se queda sin memoria, reduce `batch` (el script ya ajusta CPU/GPU).

---

## **🔎 6\. Fase 1: Detección y Organización de Resultados**

Una vez entrenado, ejecuta la fase de detección y recorte. Tienes dos opciones complementarias:

**A) Recorte automático y guardado de crops**
- Script: [detect_crop.py](detect_crop.py)
- Genera imágenes con bounding boxes y recortes individuales por clase.

```
python detect_crop.py
```

Salida:
- Imágenes anotadas: `runs/detect/result_fase_1/`
- Recortes: `runs/detect/result_fase_1/crops/sperm/`

**B) Organización por imagen con mapa numerado**
- Script: [detect_organize.py](detect_organize.py)
- Realiza la misma tarea que el script [detect_crop.py](detect_crop.py) pero este crea una carpeta por imagen en [result_fase_1](result_fase_1) y guarda:
  - Recortes numerados `*_esperma_{ID}.jpg`
  - Un “MAPA_NUMERADO” con rectángulos y etiquetas `ID`.

```
python detect_organize.py
```

Ajusta `CONFIANZA`, `CARPETA_ORIGEN` y rutas en cada script si lo necesitas.

---


## **🗺️ 7\. Roadmap Fase 2 (Clasificación/Morfología)**

- Conteo robusto y métricas por campo (confiables para reporte).
- Clasificación de morfología normal/anormal y motilidad.
- Post-procesamiento y exportación (CSV/JSON) por muestra.
- Panel simple de visualización y QA de anotaciones.

---
## **🛠️ Solución de Errores Comunes**

| Error | Causa Probable | Solución |
| :---- | :---- | :---- |
| conda: The term is not recognized | Falta PATH en Windows | Ver Sección 2.A de este documento. |
| UnauthorizedAccess / scripts disabled | PowerShell bloqueado | Ver Sección 2.B (Comando Set-ExecutionPolicy). |
| CUDA Out of Memory | La memoria de la GPU se llenó | Baja el batch=24 a batch=16 o 8 en el archivo entrenar.py. |
| Environment not found | No creaste el entorno | Ejecuta conda create... nuevamente. |
| CondaToSNonInteractiveError | Licencias no aceptadas | Ver Sección 2.C. |

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
