# Explicación Técnica: Obtención y Cálculo de la Matriz de Confusión

Este documento detalla el proceso técnico y los cálculos matemáticos utilizados para generar la matriz de confusión y las métricas de rendimiento en el proyecto de detección y clasificación de esperma.

## 1. Transformación de Imágenes a Datos Numéricos

Antes de realizar cualquier cálculo estadístico, la inteligencia artificial debe convertir las imágenes (archivos visuales) en una estructura de datos procesable por el modelo:

- **Lectura y Decodificación:** El sistema utiliza la librería `tensorflow` para leer los archivos `.jpg`. Los píxeles de color se transforman en tensores numéricos de tres dimensiones (Alto x Ancho x Canales RGB).
- **Normalización y Resize:** Cada imagen es redimensionada a una resolución estándar de **300x300 píxeles** (parámetro `IMG_SIZE`). Esto asegura que el modelo reciba siempre la misma cantidad de datos de entrada independientemente del tamaño original de la micrografía.
- **Inferencia (Propagación hacia adelante):** La imagen procesada entra en la red neuronal **EfficientNetB0**. En la capa final, se aplica una función de activación `sigmoid` para cada una de las 5 categorías de morfología.
- **Resultado de la IA:** El modelo devuelve un vector de **5 probabilidades** (valores decimales entre 0.0 y 1.0) que representan la confianza de la IA en la presencia de cada característica o defecto.

## 2. Generación de la Matriz de Confusión

La matriz de confusión es el resultado de comparar las etiquetas del **Experto (Ground Truth)** con las predicciones generadas por la **IA**.

### A. Biblioteca de Cálculo
El proceso no es un cálculo manual arbitrario, sino que utiliza estándares de la industria:
- **Librería:** `scikit-learn` (sklearn).
- **Función:** `multilabel_confusion_matrix`. Esta función permite manejar el hecho de que un solo espermatozoide puede tener múltiples defectos simultáneamente (clasificación multietiqueta). Se generan **5 matrices independientes de 2x2**, una por cada categoría.

### B. Binarización mediante Umbrales (Thresholding)
Como el modelo entrega probabilidades (ej. 0.85), el sistema debe convertirlas en una decisión binaria (0 o 1):
- El script `core/calcular_kappa.py` busca el **umbral óptimo** para cada clase de forma dinámica.
- Si la probabilidad supera el umbral (ej. 0.55), se registra como "Presencia de defecto" (1); de lo contrario, como "Ausencia" (0).

## 3. Lógica de los Cálculos Estadísticos

Para cada categoría (Normal, Cabeza Anormal, Pieza Intermedia Anormal, Cola Anormal y Residuo Citoplasmático), la matriz de confusión organiza los resultados en cuatro celdas fundamentales:

| Sigla | Nombre | Descripción |
| :--- | :--- | :--- |
| **TP** | Verdadero Positivo | El experto marcó el defecto y la IA lo detectó correctamente. |
| **TN** | Verdadero Negativo | El experto no marcó defecto y la IA confirmó que no lo había. |
| **FP** | Falso Positivo | La IA detectó un defecto que el experto no reconoció (Error tipo I). |
| **FN** | Falso Negativo | La IA no detectó un defecto que el experto sí marcó (Error tipo II). |

### Fórmulas Derivadas
A partir de estos cuatro valores, el sistema calcula automáticamente las métricas de rendimiento:

- **Exactitud (Accuracy):** $\frac{TP + TN}{Total}$. Indica el porcentaje de predicciones correctas totales.
- **Sensibilidad (Recall):** $\frac{TP}{TP + FN}$. Muestra la capacidad de la IA para encontrar todos los defectos reales.
- **Especificidad:** $\frac{TN}{TN + FP}$. Muestra la capacidad de la IA para identificar correctamente el esperma normal/sano.
- **F1-Score:** Media armónica entre precisión y sensibilidad, utilizada para medir el balance del modelo.

## 4. Flujo de Implementación en el Código

1. **Entrada:** Archivo CSV con etiquetas de expertos y carpeta de imágenes `crops`.
2. **Procesamiento:** Script `core/calcular_kappa.py` carga el modelo `.h5` y ejecuta la inferencia sobre el set de validación (10% de los datos).
3. **Cálculo:** La función `multilabel_confusion_matrix` procesa los vectores binarizados.
4. **Salida:** Generación de la tabla de contingencia y visualización final mediante `matplotlib` (archivo `confusion_matrix_word.png`).
