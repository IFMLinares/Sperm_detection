# Resultados y Salidas

## 1. Pipeline Analítico Integrado (`analizador_completo.py`)
La inferencia combinada de Detección (YOLO) y Morfología (Keras) se guarda en:
- `pruebas/resultados_20p/REPORTE_*.jpg`
- Estas imágenes contienen las detecciones recuadradas (colorizado por clase), etiquetas de diagnóstico multietiqueta (Normal, C. Anormal, P.I. Anormal, etc.) y un resumen estadístico por campo microscópico.

## 2. Validación Clínica Oficial (`calcular_kappa.py`)
Los resultados de la validación estricta para la tesis se encuentran en:
- **Archivo de Reporte:** `docs/RESULTADOS_TESIS.md`.
- **Métricas:** Tabla detallada con TP, TN, FP, FN, Sensibilidad, Especificidad, Accuracy, VPP y VPN por cada una de las 5 clases.
- **Relatividad:** Índice Kappa de Cohen (Acuerdo Aceptable ~0.33).

## 3. Entrenamiento (Keras v8)
- Los pesos finales se alojan en `models/trained/clasificacion/experimento_5/mejor_modelo_v8.h5`.
- Las gráficas de entrenamiento (Accuracy, Loss, Focal Loss) se guardan en la misma carpeta del experimento.

## Validación Manual y Ajustes
- **Umbral de Confianza:** Se puede ajustar la sensibilidad del clasificador modificando el umbral en `analizador_completo.py`.
- **Re-entrenamiento:** El modelo v8 es el resultado de la estrategia de Focal Loss; cualquier mejora futura debe mantener esta función de pérdida para manejar el desbalance.
