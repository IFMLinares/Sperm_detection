"""
preparar_dataset_morfologia.py
==============================
Genera el dataset de entrenamiento para el clasificador de morfología
(FASE 2) combinando:

1. Los crops del CSV morphology_results.csv + imágenes del roboflow_dataset
   → extrae cada bounding box individualmente usando los .txt YOLO
2. Los crops ya existentes en data/datasets/dataset_f2/normales y anormales

Salida:
    data/datasets/dataset_morfologia_v3/
    ├── normal/           (crops de espermatozoides normales)
    ├── anormal/          (crops de cualquier espermatozoide con defecto)
    └── crops_multilabel.csv  (para entrenamiento multi-label con 5 columnas)

El CSV multilabel tiene el formato:
    filepath,normal,cabeza_anormal,cola_anormal,pieza_intermedia_anormal,residuo_citoplasmatico
"""

import os
import csv
import shutil
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

# --- ⚙️ CONFIGURACIÓN ---
BASE_DIR = Path(__file__).parent.parent  # raíz del proyecto

CSV_MORFOLOGIA = BASE_DIR / "data" / "roboflow_dataset" / "morphology_results.csv"
ROBOFLOW_DIR   = BASE_DIR / "data" / "roboflow_dataset"
DATASET_F2     = BASE_DIR / "data" / "datasets" / "dataset_f2"
OUTPUT_DIR     = BASE_DIR / "data" / "datasets" / "dataset_morfologia_v3"

SPLITS = ["train", "valid", "test"]
CLASES_DEFECTO = ["cabeza_anormal", "cola_anormal", "pieza_intermedia_anormal", "residuo_citoplasmatico"]

# Padding alrededor del crop en píxeles (ayuda a que el modelo vea contexto)
CROP_PADDING = 8


def yolo_bbox_to_pixels(bbox, img_w, img_h):
    """Convierte bbox YOLO (cx, cy, w, h) normalizado a (x1, y1, x2, y2) en píxeles."""
    cx, cy, bw, bh = bbox
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return x1, y1, x2, y2


def recortar_con_padding(img, x1, y1, x2, y2, padding=CROP_PADDING):
    """Recorta un área de la imagen con padding, respetando los límites."""
    h, w = img.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    crop = img[y1:y2, x1:x2]
    return crop


def determinar_clase_binaria(row):
    """Retorna 'normal' o 'anormal' según las columnas del CSV."""
    if row["normal"] == 1:
        return "normal"
    # Cualquier defecto → anormal
    for col in CLASES_DEFECTO:
        if row[col] == 1:
            return "anormal"
    return "anormal"  # fallback: si no es normal y no tiene defecto etiquetado, tratar como anormal


def main():
    print("=" * 70)
    print("🔬 GENERADOR DE DATASET DE MORFOLOGÍA MULTI-LABEL")
    print("=" * 70)

    # Limpiar salida si existe
    if OUTPUT_DIR.exists():
        print(f"\n🗑️  Limpiando directorio existente: {OUTPUT_DIR.name}")
        shutil.rmtree(OUTPUT_DIR)

    (OUTPUT_DIR / "normal").mkdir(parents=True)
    (OUTPUT_DIR / "anormal").mkdir(parents=True)

    # Registro para el CSV multilabel
    registros_multilabel = []
    conteo = {"normal": 0, "anormal": 0, "errores": 0}

    # =========================================================================
    # PARTE 1: Procesar roboflow_dataset con el CSV de morfología
    # =========================================================================
    print("\n📋 PARTE 1: Extrayendo crops del roboflow_dataset con CSV de morfología...")

    if not CSV_MORFOLOGIA.exists():
        print(f"   ❌ CSV no encontrado: {CSV_MORFOLOGIA}")
    else:
        df = pd.read_csv(CSV_MORFOLOGIA)
        # Asegurar que las columnas son enteros
        for col in ["normal"] + CLASES_DEFECTO:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # Agrupar por imagen
        grupos = df.groupby("image_name")
        imagenes_procesadas = 0
        crops_generados = 0

        for img_nombre, grupo in grupos:
            # Buscar la imagen en los splits
            img_path = None
            label_path = None
            for split in SPLITS:
                candidato_img = ROBOFLOW_DIR / split / "images" / img_nombre
                candidato_lbl = ROBOFLOW_DIR / split / "labels" / (Path(img_nombre).stem + ".txt")
                if candidato_img.exists():
                    img_path = candidato_img
                    label_path = candidato_lbl
                    break

            if img_path is None:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                conteo["errores"] += 1
                continue

            img_h, img_w = img.shape[:2]

            # Cargar bounding boxes del archivo label YOLO
            bboxes = []
            if label_path and label_path.exists():
                lines = label_path.read_text(encoding="utf-8").strip().splitlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        bboxes.append([float(p) for p in parts[1:5]])  # cx, cy, w, h

            # Ordenar el grupo por box_index para hacer match por índice
            grupo_sorted = grupo.sort_values("box_index").reset_index(drop=True)

            for _, row in grupo_sorted.iterrows():
                box_idx = int(row["box_index"])

                if box_idx >= len(bboxes):
                    # Índice fuera de rango en el label file → saltar
                    continue

                bbox = bboxes[box_idx]
                x1, y1, x2, y2 = yolo_bbox_to_pixels(bbox, img_w, img_h)
                crop = recortar_con_padding(img, x1, y1, x2, y2)

                if crop is None or crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
                    conteo["errores"] += 1
                    continue

                clase_binaria = determinar_clase_binaria(row)
                stem = Path(img_nombre).stem
                crop_nombre = f"rf_{stem}_box{box_idx}.jpg"
                crop_path = OUTPUT_DIR / clase_binaria / crop_nombre

                cv2.imwrite(str(crop_path), crop)
                conteo[clase_binaria] += 1
                crops_generados += 1

                # Registro multilabel
                registros_multilabel.append({
                    "filepath": str(crop_path.relative_to(OUTPUT_DIR.parent.parent)),
                    "normal": int(row["normal"]),
                    "cabeza_anormal": int(row["cabeza_anormal"]),
                    "cola_anormal": int(row["cola_anormal"]),
                    "pieza_intermedia_anormal": int(row["pieza_intermedia_anormal"]),
                    "residuo_citoplasmatico": int(row["residuo_citoplasmatico"]),
                })

            imagenes_procesadas += 1

        print(f"   ✅ {imagenes_procesadas} imágenes procesadas → {crops_generados} crops generados")
        print(f"      Normal: {conteo['normal']} | Anormal: {conteo['anormal']} | Errores: {conteo['errores']}")

    # =========================================================================
    # PARTE 2: Copiar crops existentes de dataset_f2
    # =========================================================================
    print("\n📂 PARTE 2: Copiando crops existentes de dataset_f2...")

    for clase_dir_nombre, clase_destino in [("normales", "normal"), ("anormales", "anormal")]:
        origen_dir = DATASET_F2 / clase_dir_nombre
        if not origen_dir.exists():
            print(f"   ⚠️  No encontrado: {origen_dir}")
            continue

        copias = 0
        for img_path in origen_dir.iterdir():
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            nuevo_nombre = f"f2_{img_path.name}"
            destino = OUTPUT_DIR / clase_destino / nuevo_nombre
            shutil.copy2(img_path, destino)
            conteo[clase_destino] += 1
            copias += 1

            # Para los crops de dataset_f2, asignamos labels según la clase
            if clase_destino == "normal":
                registros_multilabel.append({
                    "filepath": str(destino.relative_to(OUTPUT_DIR.parent.parent)),
                    "normal": 1,
                    "cabeza_anormal": 0,
                    "cola_anormal": 0,
                    "pieza_intermedia_anormal": 0,
                    "residuo_citoplasmatico": 0,
                })
            else:
                # Para anormales de dataset_f2 sin detalle, marcamos todo como 0
                # (se usarán solo para clasificación binaria, no para el multi-label detallado)
                registros_multilabel.append({
                    "filepath": str(destino.relative_to(OUTPUT_DIR.parent.parent)),
                    "normal": 0,
                    "cabeza_anormal": 0,
                    "cola_anormal": 0,
                    "pieza_intermedia_anormal": 0,
                    "residuo_citoplasmatico": 0,
                })

        print(f"   [{clase_dir_nombre}] → {copias} crops copiados")

    # =========================================================================
    # PARTE 3: Guardar CSV multilabel
    # =========================================================================
    csv_out = OUTPUT_DIR / "crops_multilabel.csv"
    columnas = ["filepath", "normal", "cabeza_anormal", "cola_anormal",
                "pieza_intermedia_anormal", "residuo_citoplasmatico"]
    df_out = pd.DataFrame(registros_multilabel, columns=columnas)
    df_out.to_csv(csv_out, index=False, encoding="utf-8")

    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ DATASET DE MORFOLOGÍA GENERADO")
    print("=" * 70)
    total = conteo["normal"] + conteo["anormal"]
    print(f"   Normal  : {conteo['normal']:5d} crops ({conteo['normal']/total*100:.1f}%)" if total > 0 else "")
    print(f"   Anormal : {conteo['anormal']:5d} crops ({conteo['anormal']/total*100:.1f}%)" if total > 0 else "")
    print(f"   TOTAL   : {total:5d} crops")
    print(f"\n📁 Directorio : {OUTPUT_DIR}")
    print(f"📄 CSV multi-label: {csv_out}")
    print("\n🚀 Ahora puedes entrenar el clasificador con:")
    print("   python entrenar_morfologia_v3.py")


if __name__ == "__main__":
    main()
