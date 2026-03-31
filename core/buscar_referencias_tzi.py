import pandas as pd
import numpy as np
import os
import sys

# Forzar salida UTF-8 para evitar errores en Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- ⚙️ CONFIGURACIÓN ---
CSV_PATH = "data/processed/morphology_v2/metadata.csv"

# Perfiles objetivo extraídos de la captura de pantalla de Excel
TARGETS = {
    "ID_582_REF": {"norm_p": 2.0, "abn_p": 98.0, "tzi": 1.50},
    "ID_592_REF": {"norm_p": 4.0, "abn_p": 96.0, "tzi": 1.35},
    "ID_593_REF": {"norm_p": 6.0, "abn_p": 94.0, "tzi": 1.26},
    "ID_594_REF": {"norm_p": 3.0, "abn_p": 97.0, "tzi": 1.44},
    "ID_595_REF": {"norm_p": 0.0, "abn_p": 100.0, "tzi": 1.44}
}

print(f"🔍 Cargando metadatos desde {CSV_PATH}...")
if not os.path.exists(CSV_PATH):
    print(f"❌ Error: No se encontró el archivo {CSV_PATH}")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)

# Asegurar nombres de columnas (limpiar posibles espacios o caracteres extraños)
df.columns = [c.strip() for c in df.columns]

# --- 📊 AGRUPACIÓN Y CÁLCULOS ---
print("🧪 Calculando métricas por imagen original (original_image)...")

# Identificar columnas de defectos
defect_cols = ['cabeza_anormal', 'cola_anormal', 'pieza_intermedia_anormal', 'residuo_citoplasmatico']

# Agrupar por la imagen original
grouped = df.groupby('original_image').agg(
    total_cells=('normal', 'count'),
    normal_count=('normal', 'sum'),
    cabeza_anormal_count=('cabeza_anormal', 'sum'),
    cola_anormal_count=('cola_anormal', 'sum'),
    pieza_count=('pieza_intermedia_anormal', 'sum'),
    residuo_count=('residuo_citoplasmatico', 'sum')
).reset_index()

# Calcular porcentajes generales
grouped['perc_normal'] = (grouped['normal_count'] / grouped['total_cells']) * 100
grouped['perc_anormal'] = 100 - grouped['perc_normal']
grouped['abnormal_count'] = grouped['total_cells'] - grouped['normal_count']

# Calcular TZI
# TZI = Suma de todos los defectos / Número de células con al menos un defecto
grouped['sum_defects'] = (
    grouped['cabeza_anormal_count'] + 
    grouped['cola_anormal_count'] + 
    grouped['pieza_count'] + 
    grouped['residuo_count']
)
grouped['tzi'] = grouped.apply(
    lambda x: x['sum_defects'] / x['abnormal_count'] if x['abnormal_count'] > 0 else 1.0, 
    axis=1
)

# Calcular porcentajes de defectos INDIVIDUALES sobre el total de anormales (como el Excel)
grouped['perc_cabeza'] = (grouped['cabeza_anormal_count'] / grouped['abnormal_count'] * 100).fillna(0)
grouped['perc_cola'] = (grouped['cola_anormal_count'] / grouped['abnormal_count'] * 100).fillna(0)
grouped['perc_pieza'] = (grouped['pieza_count'] / grouped['abnormal_count'] * 100).fillna(0)
grouped['perc_residuo'] = (grouped['residuo_count'] / grouped['abnormal_count'] * 100).fillna(0)

# --- 🎯 BÚSQUEDA DE SIMILITUD ---
print("\n" + "="*80)
print(f"{'PERFIL EXCEL':<15} | {'MATCH ENCONTRADO':<40} | {'DIFERENCIA TZI':<10}")
print("="*80)

# Filtrar por muestras con al menos 10 células para que sea representativo
grouped_filtered = grouped[grouped['total_cells'] >= 10].copy()

if grouped_filtered.empty:
    print("⚠️ No hay imágenes con >= 10 células. Usando todo el dataset.")
    grouped_filtered = grouped.copy()

final_matches = []

for name, target in TARGETS.items():
    # Calcular puntuación de similitud
    grouped_filtered['score'] = (
        ((grouped_filtered['tzi'] - target['tzi']) ** 2) * 8 + 
        ((grouped_filtered['perc_normal'] - target['norm_p']) ** 2) * 1
    )
    
    # Obtener el mejor match
    best = grouped_filtered.loc[grouped_filtered['score'].idxmin()]
    
    diff_tzi = abs(best['tzi'] - target['tzi'])
    
    print(f"{name:<15} | {best['original_image'][:40]:<40} | {diff_tzi:.4f}")
    
    final_matches.append({
        "target": name,
        "match_id": best['original_image'],
        "tzi_found": best['tzi'],
        "norm_found": best['perc_normal'],
        "abn_found": best['perc_anormal'],
        "head_found": best['perc_cabeza'],
        "tail_found": best['perc_cola'],
        "mid_found": best['perc_pieza'],
        "res_found": best['perc_residuo'],
        "total_cells": best['total_cells']
    })

# --- 📝 GUARDAR RESUMEN ---
summary_path = "docs/REPORTES_EQUIVALENTES.md"
os.makedirs("docs", exist_ok=True)

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("# Reporte de Equivalencias Morfológicas (IA vs Excel)\n\n")
    f.write("Este reporte identifica las imágenes que mejor representan los perfiles de teratozoospermia del Excel.\n\n")
    f.write("| Referencia | Imagen Original Match | % Norm | % Abn | % Cab | % Col | % Pza | % Res | TZI |\n")
    f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for m in final_matches:
        f.write(f"| {m['target']} | `{m['match_id'][:35]}...` | {m['norm_found']:.1f}% | {m['abn_found']:.1f}% | {m['head_found']:.1f}% | {m['tail_found']:.1f}% | {m['mid_found']:.1f}% | {m['res_found']:.1f}% | **{m['tzi_found']:.2f}** |\n")

    f.write("\n\n*Nota: Los porcentajes de defectos (Cab, Col, Pza, Res) se calculan sobre el total de espermatozoides anormales.*")

print(f"\n✅ Reporte generado en: {summary_path}")
