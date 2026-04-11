import requests
import pandas as pd
import os
import time
import json
import logging

PROCESSED_PATH = os.path.join("data", "processed", "variants.parquet")
ANNOTATED_PATH = os.path.join("data", "processed", "annotated.parquet")

VEP_URL = "https://rest.ensembl.org/vep/human/region"

BATCH_SIZE = 200
SAMPLE_SIZE = 50000
SLEEP_BETWEEN = 1.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def sample_variants(df: pd.DataFrame, n: int = SAMPLE_SIZE) -> pd.DataFrame:
    chrom_proportion = df['chrom'].value_counts(normalize=True)
    sample_chrom_counts = (chrom_proportion * n).round().astype(int)

    sample_chrom_counts = sample_chrom_counts.clip(lower=1)

    sampled_dfs = []
    for chrom, count in sample_chrom_counts.items():
        chrom_df = df[df['chrom'] == chrom]
        if len(chrom_df) <= count:
            sampled_dfs.append(chrom_df)
        else:
            sampled_dfs.append(chrom_df.sample(count, random_state=42))

    logging.info(f"Sampled {sum(len(df) for df in sampled_dfs):,} variants across {len(sampled_dfs)} chromosomes.") 
    return pd.concat(sampled_dfs, ignore_index=True)

def format_variant(row) -> str:
    return f"{row.chrom} {row.pos} {row.pos} {row.ref}/{row.alt} 1"

def query_vep(batch) -> list:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    body = json.dumps({"variants": batch})
    response = requests.post(VEP_URL, headers=headers, data=body, timeout=30)
    response.raise_for_status()
    return response.json()

def parse_result(result: dict) -> dict:
    _input = result.get("input", "").split()
    chrom = _input[0]
    pos = _input[1]
    ref_alt = _input[3]
    ref, alt = ref_alt.split("/")   

    consequence = result.get("most_severe_consequence", "unknown")

    tc = result.get("transcript_consequences", [{}])[0]

    gene_id = tc.get("gene_id", "unknown")
    gene_symbol = tc.get("gene_symbol", "unknown")
    impact = tc.get("impact", "unknown")  

    colocated = result.get("colocated_variants", [])
    if colocated:
        rs_id = colocated[0].get("id", "novel")
    else:
        rs_id = "novel"

    return {
        "chrom": chrom,
        "pos": int(pos),
        "ref": ref,
        "alt": alt,
        "consequence": consequence,
        "gene_id": gene_id,
        "gene_symbol": gene_symbol,
        "impact": impact,
        "rs_id": rs_id
    }

def annotate_variants(df: pd.DataFrame) -> pd.DataFrame:
    formatted = [format_variant(row) for row in df.itertuples()]

    batches = [formatted[i:i+BATCH_SIZE] for i in range(0, len(formatted), BATCH_SIZE)]

    logging.info(f"Annotating {len(df):,} variants in {len(batches)} batches...")

    done = set()

    if os.path.exists(ANNOTATED_PATH):
        _df = pd.read_parquet(ANNOTATED_PATH)
        done = set(zip(_df['chrom'], _df['pos']))
    
    all_records = []

    for i, batch in enumerate(batches):
        try:
            results = query_vep(batch)
            for result in results:
                record = parse_result(result)
                if (record['chrom'], record['pos']) not in done:
                    all_records.append(record)
        except requests.exceptions.RequestException as e:    
            logging.warning(f"Batch {i} failed: {e}, skipping...")
            time.sleep(5)
            continue

        time.sleep(SLEEP_BETWEEN)

        if i % 50 == 0 and i > 0:
            pd.DataFrame(all_records).to_parquet(ANNOTATED_PATH, index=False)
            logging.info(f"Checkpoint saved at batch {i} ({len(all_records):,} variants)")

    annotated_df = pd.DataFrame(all_records)
    annotated_df.to_parquet(ANNOTATED_PATH, index=False)
    logging.info(f"Annotation complete. Total annotated variants: {len(annotated_df):,}")
    return annotated_df

if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_PATH)
    sampled = sample_variants(df)
    annotated = annotate_variants(sampled)
    print(annotated.shape)
    print(annotated["consequence"].value_counts())