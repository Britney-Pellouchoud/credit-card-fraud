import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def load_dataset(path):
    print("START RUN")

    full_path = ROOT / path
    df = pd.read_csv(full_path)

    # paper dataset: fraud label column is "Class"
    if "Class" not in df.columns:
        raise ValueError(f"Missing 'Class' column. Found: {df.columns}")

    print("LOADING DATA")
    X = df.drop(columns=["Class"])
    y = df["Class"]
    print("DATA LOADED")

    return X, y