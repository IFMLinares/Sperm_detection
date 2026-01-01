# Troubleshooting (Solución de problemas)

## GPU/Entorno
- `python check_gpu.py` para versiones de Python/PyTorch y estado de CUDA.
- Si `torch.cuda.is_available()` es `False`, instala PyTorch con CUDA.

## Conda/PowerShell
- Agrega Conda al PATH y ejecuta `conda init powershell`.
- `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` si hay bloqueos.

## Memoria GPU
- Baja `batch` en `train_hunter.py`.
- Cierra procesos que usen la GPU.

## Rutas/Archivos
- Verifica `data.yaml` antes de entrenar.
- Asegúrate de poner imágenes en `my_images/` antes de detectar.
