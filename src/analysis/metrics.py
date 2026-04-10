import os
import pandas as pd
import json

PROCCESSED_PATH = os.path.join("data", "processed", "variants.parquet")

def load_data(path) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data file not found at {path}")
    df = pd.read_parquet(path)
    print(df.shape)
    print(df.columns)
    return df
