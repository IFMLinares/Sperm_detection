import pandas as pd
from pathlib import Path

CSV_PATH = Path(r"e:\Programs Installed\xampp\htdocs\IA\SpermDetection\data\datasets\dataset_morfologia_v3\crops_multilabel.csv")
if not CSV_PATH.exists():
    print(f"File not found: {CSV_PATH}")
    exit(1)

df = pd.read_csv(CSV_PATH)
CLASES = ["normal", "cabeza_anormal", "cola_anormal", "pieza_intermedia_anormal", "residuo_citoplasmatico"]

print(f"Total rows: {len(df)}")
for c in CLASES:
    count = df[c].sum()
    print(f"  {c:30}: {count} ({count/len(df)*100:.2f}%)")
