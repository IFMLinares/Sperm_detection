# Preguntas Frecuentes (FAQ)

**¿No detecta mi GPU?**
- Ejecuta `python check_gpu.py`.
- Reinstala PyTorch con CUDA adecuado.

**¿Sin recortes o pocos hallazgos?**
- Sube `CONFIANZA` si hay ruido; bájala si falta detección.
- Revisa calidad del dataset y foco del microscopio.

**¿Descarga fallida de Roboflow?**
- Verifica Internet y API Key.
- Reintenta y revisa cambios de workspace/proyecto.
