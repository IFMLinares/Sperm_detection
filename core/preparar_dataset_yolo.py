"""
preparar_dataset_yolo.py
========================
Combina el dataset propio (roboflow_dataset) con el dataset externo (Sperm-Detection-2)
en un único dataset YOLO unificado con una sola clase: 'esperma' (clase 0).

El roboflow_dataset usa:
  - clase 0: 'esperma' (anormal)
  - clase 1: 'sperm_normal'

Sperm-Detection-2 usa:
  - clase 0: espermatozoide (una sola clase)

Al unificar, todas las detecciones serán clase 0 (esperma), ya que solo
queremos detectar la presencia de espermatozoides. La clasificación morfológica
la hace el modelo de FASE 2.

Salida: data/datasets/dataset_yolo_unificado/
"""

import os
import shutil
import yaml
from pathlib import Path

# --- ⚙️ CONFIGURACIÓN ---
BASE_DIR = Path(__file__).parent.parent  # raíz del proyecto

# Datasets de entrada
DATASET_PROPIO = BASE_DIR / "data" / "roboflow_dataset"       # Tu dataset
DATASET_EXTERNO = BASE_DIR / "data" / "datasets" / "Sperm-Detection-2"  # Dataset público

# Salida
OUTPUT_DIR = BASE_DIR / "data" / "datasets" / "dataset_yolo_unificado"

SPLITS = ["train", "valid", "test"]


def unificar_label(txt_origen: Path, txt_destino: Path, clase_a_unificar: bool = True):
    """
    Copia un archivo de labels YOLO al destino, asegurando que todas las
    clases queden como 0 (esperma unificado).
    """
    if not txt_origen.exists():
        return 0
    lines = txt_origen.read_text(encoding="utf-8").strip().splitlines()
    nuevas_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            # Forzar clase 0 independientemente de la clase original
            parts[0] = "0"
            nuevas_lines.append(" ".join(parts))
    txt_destino.parent.mkdir(parents=True, exist_ok=True)
    txt_destino.write_text("\n".join(nuevas_lines), encoding="utf-8")
    return len(nuevas_lines)


def copiar_split(dataset_dir: Path, split: str, output_split_dir: Path, prefijo: str = "") -> dict:
    """
    Copia imágenes y labels de un split específico al directorio de salida.
    Retorna estadísticas de lo copiado.
    """
    img_dir_origen = dataset_dir / split / "images"
    lbl_dir_origen = dataset_dir / split / "labels"

    img_dir_destino = output_split_dir / "images"
    lbl_dir_destino = output_split_dir / "labels"

    img_dir_destino.mkdir(parents=True, exist_ok=True)
    lbl_dir_destino.mkdir(parents=True, exist_ok=True)

    if not img_dir_origen.exists():
        print(f"   ⚠️  No existe: {img_dir_origen}")
        return {"imagenes": 0, "labels": 0}

    imagenes_copiadas = 0
    labels_copiadas = 0

    for img_path in img_dir_origen.iterdir():
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        # Nombre con prefijo para evitar colisiones entre datasets
        nuevo_nombre = f"{prefijo}{img_path.name}" if prefijo else img_path.name
        destino_img = img_dir_destino / nuevo_nombre
        shutil.copy2(img_path, destino_img)
        imagenes_copiadas += 1

        # Buscar el label correspondiente
        label_nombre = img_path.stem + ".txt"
        label_origen = lbl_dir_origen / label_nombre
        label_destino = lbl_dir_destino / (Path(nuevo_nombre).stem + ".txt")

        count = unificar_label(label_origen, label_destino)
        if count > 0:
            labels_copiadas += 1

    return {"imagenes": imagenes_copiadas, "labels": labels_copiadas}


def main():
    print("=" * 60)
    print("🔗 UNIFICADOR DE DATASETS YOLO")
    print("=" * 60)

    # Limpiar salida si existe
    if OUTPUT_DIR.exists():
        print(f"\n🗑️  Limpiando directorio existente: {OUTPUT_DIR.name}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    totales = {split: {"imagenes": 0, "labels": 0} for split in SPLITS}

    # --- 1. Procesar Dataset Propio ---
    print(f"\n📂 Procesando dataset PROPIO: {DATASET_PROPIO.name}")
    if not DATASET_PROPIO.exists():
        print(f"   ❌ No encontrado: {DATASET_PROPIO}")
    else:
        for split in SPLITS:
            stats = copiar_split(
                DATASET_PROPIO,
                split,
                OUTPUT_DIR / split,
                prefijo="propio_"
            )
            totales[split]["imagenes"] += stats["imagenes"]
            totales[split]["labels"] += stats["labels"]
            print(f"   [{split}] → {stats['imagenes']} imágenes, {stats['labels']} labels")

    # --- 2. Procesar Dataset Externo ---
    print(f"\n📂 Procesando dataset EXTERNO: {DATASET_EXTERNO.name}")
    if not DATASET_EXTERNO.exists():
        print(f"   ❌ No encontrado: {DATASET_EXTERNO}")
    else:
        for split in SPLITS:
            stats = copiar_split(
                DATASET_EXTERNO,
                split,
                OUTPUT_DIR / split,
                prefijo="ext_"
            )
            totales[split]["imagenes"] += stats["imagenes"]
            totales[split]["labels"] += stats["labels"]
            print(f"   [{split}] → {stats['imagenes']} imágenes, {stats['labels']} labels")

    # --- 3. Generar data.yaml unificado ---
    yaml_content = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": 1,
        "names": ["esperma"]
    }
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    # --- 4. Resumen ---
    print("\n" + "=" * 60)
    print("✅ DATASET UNIFICADO GENERADO")
    print("=" * 60)
    total_imgs = 0
    for split in SPLITS:
        imgs = totales[split]["imagenes"]
        total_imgs += imgs
        print(f"   {split:6s}: {imgs:4d} imágenes")
    print(f"   {'TOTAL':6s}: {total_imgs:4d} imágenes")
    print(f"\n📁 Directorio: {OUTPUT_DIR}")
    print(f"📄 Config:     {yaml_path}")
    print("\n🚀 Ahora puedes entrenar YOLO con:")
    print("   python train_hunter.py")
    print("   (Asegúrate de que apunte al data.yaml del dataset_yolo_unificado)")


if __name__ == "__main__":
    main()
