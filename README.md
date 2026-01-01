# **🧬 Detección de Espermatozoides con IA (Fase 1: Detección)**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-3776AB)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Este proyecto implementa un sistema de Visión Artificial basado en los estándares del **Manual de la OMS (2021)** para el análisis de semen.

**Objetivo de la Fase 1:** Detectar y recortar automáticamente espermatozoides individuales en imágenes de microscopía (100x) utilizando **YOLOv8** y aceleración por GPU (NVIDIA RTX 3080).

## **📋 1\. Requisitos Previos**

Antes de comenzar, asegúrate de tener instalado:

1. **Miniconda** (Gestor de entornos Python).  
2. **Visual Studio Code** (Editor de código).  
3. **Drivers de NVIDIA** actualizados (para soporte CUDA).  
4. **Hardware Recomendado:** GPU NVIDIA con al menos 4GB de VRAM (Probado en RTX 3080 10GB).

## **⚙️ 2\. Configuración Inicial del Sistema (Solo primera vez)**

Si es la primera vez que usas Conda en Windows con VS Code, debes realizar estos ajustes críticos para evitar errores de permisos, rutas y licencias.

### **A. Agregar Conda al PATH de Windows**

Si la terminal no reconoce el comando conda, debes agregarlo manualmente a las variables de entorno:

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

*⚠️ **IMPORTANTE:** Cierra la terminal actual (icono de basura 🗑️) y abre una nueva para aplicar estos cambios.*

### **C. Aceptar Términos de Servicio (Solución a error 'CondaToSNonInteractiveError')**

Anaconda requiere aceptar explícitamente las licencias de sus canales. Ejecuta estos comandos si recibes errores al crear el entorno:

conda tos accept \--override-channels \--channel \[https://repo.anaconda.com/pkgs/main\](https://repo.anaconda.com/pkgs/main)  
conda tos accept \--override-channels \--channel \[https://repo.anaconda.com/pkgs/r\](https://repo.anaconda.com/pkgs/r)  
conda tos accept \--override-channels \--channel \[https://repo.anaconda.com/pkgs/msys2\](https://repo.anaconda.com/pkgs/msys2)

## **🚀 3\. Creación del Entorno Virtual**

Sigue estos pasos para crear el entorno aislado y cargar las librerías de Inteligencia Artificial:

\# 1\. Crear el entorno (Python 3.10 es la versión más estable para YOLOv8)  
conda create \-n tesis\_espermas python=3.10 \-y

\# 2\. Activar el entorno  
conda activate tesis\_espermas  
\# (IMPORTANTE: Debes ver '(tesis\_espermas)' en verde a la izquierda de tu terminal)

\# 3\. Instalar librerías principales  
pip install ultralytics roboflow

## **📥 4\. Descarga del Dataset (Script existente)**

Usa el script ya incluido [download_data.py](download_data.py) para descargar automáticamente el dataset etiquetado desde Roboflow Universe y organizarlo en formato YOLOv8.

- El script requiere tu API Key privada de Roboflow. Evita publicar esa clave en GitHub.
- Si cambiaste credenciales, edita temporalmente el archivo y considera mover la clave a una variable de entorno.

**Ejecutar descarga:**

```
python download_data.py
```

**Nota de seguridad:** No subas tu API Key al repositorio público. En producción, usa variables de entorno (`set ROBOFLOW_API_KEY=...`) y lee `os.getenv` en el script.

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
- Crea una carpeta por imagen en [result_fase_1](result_fase_1) y guarda:
  - Recortes numerados `*_esperma_{ID}.jpg`
  - Un “MAPA_NUMERADO” con rectángulos y etiquetas `ID`.

```
python detect_organize.py
```

Ajusta `CONFIANZA`, `CARPETA_ORIGEN` y rutas en cada script si lo necesitas.

---

## **🛡️ 7\. Seguridad y Buenas Prácticas**

- **API Keys:** No publiques credenciales. Usa variables de entorno o `.env`.
- **Pesos del modelo:** Verifica licencias si subes `*.pt` a GitHub.
- **Datos sensibles:** Anonimiza imágenes si son clínicas.

**Configurar `.env` (recomendado):**
- Copia [\.env.example](.env.example) a `.env` y rellena `ROBOFLOW_API_KEY`.
- En Windows también puedes definirla de forma persistente:

```
setx ROBOFLOW_API_KEY TU_API_KEY
```

El script [download_data.py](download_data.py) lee `ROBOFLOW_API_KEY` desde el entorno o `.env`.

---

## **🗺️ 8\. Roadmap Fase 2 (Clasificación/Morfología)**

- Conteo robusto y métricas por campo (confiables para reporte).
- Clasificación de morfología normal/anormal y motilidad.
- Post-procesamiento y exportación (CSV/JSON) por muestra.
- Panel simple de visualización y QA de anotaciones.

---

## **📤 9\. Publicación en GitHub: qué subir y qué excluir**

- **Licencia:**
  - **MIT**: muy simple y permisiva; ideal si quieres facilitar uso y reutilización sin condiciones complejas.
  - **Apache-2.0**: también permisiva, con cláusula explícita de patentes y términos más detallados; útil si te interesa cobertura legal adicional.
  - Elige MIT por simplicidad; Apache si te importa la protección/claridad de patentes.

- **Secretos:** no subas credenciales. Usa `.env` (ejemplo en [\.env.example](.env.example)).

- **Datos locales (imágenes originales):**
  - Si son sensibles (clínicos), **no publiques**. Considera subir solo **un pequeño subconjunto anonimizado** o recortado.
  - Para datasets grandes, usa **Git LFS** o enlaces externos (Roboflow, Kaggle, Drive) y documenta cómo obtenerlos.

- **Resultados (crops, mapas numerados):**
  - No es necesario subir todo; incluye **unos pocos ejemplos** en `documentation/` o como imágenes en el README.
  - El resto queda excluido por [\.gitignore](.gitignore): `result_fase_1/` y `runs/`.

- **Pesos del modelo (`*.pt`):**
  - Tus pesos entrenados pueden ser grandes; compártelos como **release assets**, en **Hugging Face**, **Drive**, o con **Git LFS**.
  - Verifica licencias del modelo base y dataset antes de distribuir pesos.

El repositorio ya incluye [\.gitignore](.gitignore) para excluir `.env`, `my_images/`, `result_fase_1/`, `runs/`.

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
