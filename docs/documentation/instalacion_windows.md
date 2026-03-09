# Instalación y Configuración en Windows

## 1. Requisitos previos
- Miniconda o Anaconda (se recomienda Miniconda).
- Visual Studio Code.
- Drivers NVIDIA actualizados (si usarás GPU/CUDA).

## 2. Agregar Conda al PATH (si no se reconoce `conda`)
1. Abre "Editar las variables de entorno de esta cuenta".
2. En `Path`, agrega (ajusta `TU_USUARIO`):
   - `C:\Users\TU_USUARIO\miniconda3`
   - `C:\Users\TU_USUARIO\miniconda3\Scripts`
   - `C:\Users\TU_USUARIO\miniconda3\Library\bin`
3. Reinicia VS Code.

## 3. Inicializar PowerShell y permisos
En una terminal de VS Code:
```
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
conda init powershell
```
Cierra la terminal y abre una nueva.

## 4. Aceptar Términos de Servicio (CondaToSNonInteractiveError)
Si aparecen errores al crear el entorno:
```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

## 5. Crear entorno y dependencias
```
conda create -n tesis_espermas python=3.10 -y
conda activate tesis_espermas
pip install -r requirements.txt
```

## 6. Variables de entorno / .env
- Crea `.env` (o usa [../.env.example](../.env.example)) con `ROBOFLOW_API_KEY`.
- Alternativa persistente en Windows:
```
setx ROBOFLOW_API_KEY TU_API_KEY
```
