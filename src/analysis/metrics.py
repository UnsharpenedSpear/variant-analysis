import os
import pandas as pd
import json

PROCESSED_PATH = os.path.join("data", "processed", "variants.parquet")
METRICS_PATH = os.path.join("results", "metrics.json")

def load_data(path) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data file not found at {path}")
    df = pd.read_parquet(path)
    print(f"Loaded {df.shape[0]:,} variants.")
    return df

def compute_titv(df: pd.DataFrame) -> dict:

    snps = df[df['var_type'] == 'snp'].copy()
    

    transitions = (
        ((snps['ref'] == 'A') & (snps['alt'] == 'G')) |
        ((snps['ref'] == 'G') & (snps['alt'] == 'A')) |
        ((snps['ref'] == 'C') & (snps['alt'] == 'T')) |
        ((snps['ref'] == 'T') & (snps['alt'] == 'C'))
    )
    
    ti_count = transitions.sum()
    tv_count = len(snps) - ti_count
    titv_ratio = ti_count / tv_count if tv_count > 0 else 0.0
    
    print(f"Ti/Tv Ratio: {titv_ratio:.3f} ({ti_count:,} Ti / {tv_count:,} Tv)")
    return {
        "ti_count": int(ti_count),
        "tv_count": int(tv_count),
        "titv_ratio": float(round(titv_ratio, 3))
    }

def compute_variant_types(df: pd.DataFrame) -> dict:
    counts = df['var_type'].value_counts().to_dict()

    pcts = {k: float(round((v / len(df)) * 100, 2)) for k, v in counts.items()}
    
    print("Variant Types:", pcts)
    return {"counts": counts, "percentages": pcts}

def compute_chrom_distribution(df: pd.DataFrame) -> dict:

    order = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
    
    dist = df['chrom'].value_counts().reindex(order).dropna().astype(int).to_dict()
    print("Top Chromosomes:", sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3])
    return dist

def compute_depth_stats(df: pd.DataFrame) -> dict:

    depths = df['dp'].dropna().astype(float)
    
    stats = {
        "mean": float(round(depths.mean(), 2)),
        "median": float(depths.median()),
        "std": float(round(depths.std(), 2)),
        "low_dp_pct": float(round((depths < 10).mean() * 100, 2))
    }
    print("Depth Stats:", stats)
    return stats

if __name__ == "__main__":
    data = load_data(PROCESSED_PATH)
    
    results = {
        "titv": compute_titv(data),
        "variant_types": compute_variant_types(data),
        "chrom_distribution": compute_chrom_distribution(data),
        "depth_stats": compute_depth_stats(data)
    }
    
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nAll metrics saved to {METRICS_PATH}")                  