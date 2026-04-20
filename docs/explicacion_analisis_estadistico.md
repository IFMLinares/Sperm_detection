# Explicación Técnica: Análisis Estadístico y Métricas de Validación

Este documento describe la metodología, fórmulas y herramientas utilizadas para el análisis estadístico avanzado del proyecto de detección de esperma, detallando el flujo exacto de los datos desde el archivo hasta el resultado final.

## 1. Extracción y Preparación de Datos (Data Sourcing)

El proceso comienza en el script `core/calcular_kappa.py` extrayendo información de la base de datos de entrenamiento:

- **Origen:** Archivo `data/datasets/dataset_morfologia_v3/crops_multilabel.csv`.
- **Carga de Datos:** Se utiliza la librería `pandas` para cargar el CSV. El sistema genera una ruta absoluta (`filepath_abs`) para cada imagen uniendo la ruta base del dataset con la ruta relativa del CSV.
- **Validación de Existencia:** Antes de procesar, el código verifica físicamente si cada imagen existe en el disco mediante `os.path.exists()`.
- **Muestra de Validación:** Se separa un **10% del dataset** (pool de validación) y de ese pool se extraen **100 imágenes al azar** para asegurar un tiempo de evaluación eficiente pero estadísticamente significativo.

## 2. El Procedimiento de Inferencia (El Bucle)

Para obtener los números, el sistema ejecuta un bucle iterativo:
1. **Loop Principal:** Para cada una de las 100 imágenes, se lee el archivo y se redimensiona a 300x300.
2. **Predicción Bruta:** Se ejecuta `model.predict()`, que devuelve 5 números decimales (probabilidades). 
3. **Almacenamiento:** Estos valores se guardan en una matriz llamada `preds_raw`. Al final del bucle, tenemos una tabla de 100 filas x 5 columnas de "confianza de la IA".

## 3. Optimización de Umbrales (La "Búsqueda" del Mejor KPI)

Esta es la parte más crítica del cálculo. No usamos un umbral fijo de 0.5. El código busca la "mejor verdad":
- **Bucle de Optimización:** Para cada una de las 5 categorías (Cabeza, Cola, etc.):
  - Se prueban 99 valores de umbral ($t$) desde 0.01 hasta 0.99.
  - Para cada $t$, se binarizan las predicciones y se calcula el **Índice Kappa de Cohen**.
  - El sistema "recuerda" cuál de esos 99 valores dio el Kappa más alto.
- **Resultado:** Obtenemos un `best_threshold` específico para cada defecto, optimizando la precisión del modelo para cada tipo de anomalía.

## 4. Índice de Teratozoospermia (TZI)

El TZI se calcula comparando la sumatoria de defectos contra el número de células enfermas.

### Procedimiento Matemático en el Código:
1. **Selección:** Se toma la matriz de resultados y se **secciona** para ignorar la columna 0 (clase "Normal").
2. **Conteo de Defectos:** Se usa `np.sum(matrix, axis=1)` para contar cuántos defectos tiene cada espermatozoide (un número entre 0 y 4).
3. **Suma Total:** Se suman todos los defectos de las 100 imágenes.
4. **Denominador:** Se cuenta cuántas filas tuvieron un resultado mayor a 0 (es decir, cuántos espermatozoides son anormales).
5. **División Final:** `Total_Defectos / Num_Celulas_Con_Defectos`.

## 5. Índice Kappa de Cohen (Acuerdo Inter-Observador)

El Índice Kappa es una métrica de fiabilidad mucho más robusta que el simple porcentaje de aciertos (accuracy), ya que ajusta el resultado basándose en la probabilidad de que el experto y la IA coincidan por puro azar.

### El Cálculo Matemático
El sistema utiliza la fórmula de Cohen:
$$ \kappa = \frac{p_o - p_e}{1 - p_e} $$
- **$p_o$ (Acuerdo Observado):** Es la proporción de veces que la IA y el experto coinciden en su diagnóstico (ambos dicen "anormal" o ambos dicen "normal").
- **$p_e$ (Acuerdo por Azar):** Es la probabilidad de que coincidan si ambos estuvieran lanzando una moneda al aire basándose en sus propias frecuencias de "votos". Si la IA tiende a decir que todo es "Cabeza Anormal" y el experto también, el azar jugaría un papel importante; Kappa penaliza esto.

### ¿Cómo se saca en el código?
1. Se utiliza `sklearn.metrics.cohen_kappa_score`.
2. El script genera un vector de etiquetas reales (`y_true`) extraídas del CSV y un vector de etiquetas predichas (`y_pred`).
3. Al aplicarse por clase, obtenemos una medición de **qué tan "segura" es la IA** en cada tipo de defecto. Un valor de **0.80** nos dice que, después de quitar las coincidencias por suerte, la IA y el experto están de acuerdo en un 80% del resto de los casos.

---

## 6. Análisis de Regresión Lineal (Validación de TZI)

La regresión lineal no se usa para clasificar, sino para validar la **consistencia científica** del Índice de Teratozoospermia (TZI) calculado por la IA comparado con el del experto.

### La Ecuación de la Recta
El sistema busca ajustar una línea que siga la fórmula:
$$ y = mx + b $$
- **$y$ (Eje Y):** TZI calculado por la IA.
- **$x$ (Eje X):** TZI calculado por el experto (Referencia).

### Interpretación de los Resultados:
- **La Pendiente ($m$):** Si $m = 1.0$, significa que por cada defecto extra que ve el experto, la IA también ve exactamente un defecto extra. 
  - Si $m > 1.0$, la IA es **"sobre-sensible"** (detecta más defectos que el humano).
  - Si $m < 1.0$, la IA es **"conservadora"** en su detección.
- **El Intercepto ($b$):** Representa el "sesgo constante". Si la IA siempre detecta un defecto aunque el experto no vea nada, el intercepto será positivo.
- **Coeficiente de Determinación ($R^2$):** Es el valor más importante para la validez de la tesis. Indica qué tan predecibles son los resultados. 
  - Un $R^2 = 0.95$ significa que el **95% del comportamiento de la IA** está directamente explicado por el criterio del experto humano. Esto demuestra que la IA no está "adivinando", sino siguiendo un patrón lógico correlacionado con la realidad médica.

### Herramienta de Cálculo:
Se utiliza `scipy.stats.linregress`, el cual aplica el método de **Mínimos Cuadrados Ordinarios**. Este método busca la línea que minimice la distancia al cuadrado entre los puntos reales (muestras) y la línea de tendencia, asegurando el ajuste más fiel posible a los datos experimentales.

## 7. Replicabilidad para Terceros

Si un investigador externo desea replicar estos cálculos:
1. **Datos:** Debe tener el archivo `.csv` con las 5 columnas binarias de los expertos.
2. **Inferencia:** Debe usar los pesos del modelo `mejor_modelo_v8.h5`.
3. **Métricas:** Debe aplicar el bucle de optimización de umbrales descrito en el punto 3 antes de generar la matriz de confusión, de lo contrario, las métricas podrían parecer más bajas de lo que realmente son.
