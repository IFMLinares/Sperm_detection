import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
import cv2

# Configuración de estilo "Normal para Word"
plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})

def generate_confusion_matrix():
    # Datos de la Tabla 8 (Validación v8)
    # Formato: [TP, TN, FP, FN]
    data = {
        "Normal": [3, 87, 6, 4],
        "Cabeza Anormal": [69, 12, 12, 7],
        "Cola Anormal": [52, 19, 18, 11],
        "P. Intermedia": [59, 11, 28, 2],
        "Res. Citoplas." : [6, 74, 9, 11]
    }
    
    classes = list(data.keys())
    # Vamos a crear una visualización representativa de la matriz global o una por clase
    # Para Word, una tabla visual de TP/TN/FP/FN es muy útil.
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [["Categoría", "TP", "TN", "FP", "FN", "Accuracy"]]
    for cls in classes:
        tp, tn, fp, fn = data[cls]
        acc = (tp + tn) / (tp + tn + fp + fn)
        table_data.append([cls, tp, tn, fp, fn, f"{acc:.2f}"])
        
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title("Tabla de Contingencia (Matriz de Confusión) - Validación IA v8", pad=20)
    plt.savefig("confusion_matrix_word.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Matriz de confusión guardada.")

def generate_regression():
    # Datos de TZI (Experto vs IA) de las 5 muestras
    # Experto (Target): 1.50, 1.35, 1.26, 1.44, 1.44
    # IA (Match): 1.57, 2.11, 1.87, 2.11, 1.57
    expert_tzi = np.array([1.50, 1.35, 1.26, 1.44, 1.44])
    ia_tzi = np.array([1.57, 2.11, 1.87, 2.11, 1.57])
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(expert_tzi, ia_tzi)
    r2 = r_value**2
    
    plt.figure(figsize=(6, 5))
    plt.scatter(expert_tzi, ia_tzi, color='blue', label='Muestras (n=5)')
    
    # Línea de regresión
    line = slope * expert_tzi + intercept
    plt.plot(expert_tzi, line, color='red', linestyle='--', label=f'Regresión (R²={r2:.4f})')
    
    plt.xlabel("TZI Experto (Referencia)")
    plt.ylabel("TZI IA (Antigravity v8)")
    plt.title("Análisis de Regresión Lineal - Índice de Teratozoospermia")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Añadir texto con la ecuación
    plt.text(1.28, 2.05, f"y = {slope:.4f}x + {intercept:.4f}\nR² = {r2:.4f}", 
             bbox=dict(facecolor='white', alpha=0.5))
    
    plt.savefig("regression_plot_word.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Regresión lineal guardada. R2={r2:.4f}")

def generate_augmentation_demo():
    # Ruta de una imagen de muestra
    img_path = r"e:\Programs_Installed\xampp\htdocs\IA\SpermDetection\data\datasets\dataset_f2\normales\IMG_20251117_151026_esperma_0.jpg"
    
    if not os.path.exists(img_path):
        print("⚠️ Imagen de muestra no encontrada para augmentation demo.")
        return
        
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Simulamos las transformaciones del script entrenar_morfologia.py
    # 1. Flip
    img_flip = cv2.flip(img, 1)
    # 2. Brillo (random_brightness 0.2)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.2, 0, 255)
    img_bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # 3. Rotación (rot90)
    img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(img); axes[0].set_title("Original")
    axes[1].imshow(img_flip); axes[1].set_title("Flip Horizontal")
    axes[2].imshow(img_bright); axes[2].set_title("Brillo (+20%)")
    axes[3].imshow(img_rot); axes[3].set_title("Rotación 90°")
    
    for ax in axes:
        ax.axis('off')
        
    plt.suptitle("Técnicas de Data Augmentation Aplicadas (Fase 2)", y=1.05)
    plt.savefig("augmentation_demo_word.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Demo de augmentation guardada.")

if __name__ == "__main__":
    try:
        generate_confusion_matrix()
        generate_regression()
        generate_augmentation_demo()
        print("\n🚀 Todos los visuales han sido generados exitosamente.")
    except Exception as e:
        print(f"❌ Error durante la generación: {e}")
