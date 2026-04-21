import requests
import pandas as pd
import os
import time
import json
import logging

PROCESSED_PATH = os.path.join("data", "processed", "clinvar.parquet")
ANNOTATED_PATH = os.path.join("data", "processed", "clinvar_annotated.parquet")

VEP_URL = "https://rest.ensembl.org/vep/human/region"

BATCH_SIZE = 50
SLEEP_BETWEEN = 2.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def format_variant(row) -> str:
    return f"{row.chrom} {row.pos} {row.pos} {row.ref}/{row.alt} 1"

def query_vep(batch) -> list:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    body = json.dumps({"variants": batch})
    response = requests.post(VEP_URL, headers=headers, data=body, timeout=60)
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
        except (requests.exceptions.RequestException, KeyboardInterrupt) as e:
            logging.warning(f"Batch {i} failed: {e}, skipping...")

            if "NameResolutionError" in str(e) or "getaddrinfo" in str(e):
                logging.warning("Internet appears down, waiting 60 seconds...")
                time.sleep(60)
            else:
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
    
    pathogenic = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
    benign = {"Benign", "Likely_benign", "Benign/Likely_benign"}
    
    df = df.dropna(subset=["ref", "alt"])
    df = df[df["alt"].str.match(r'^[ACGT]+$', na=False)]
    df = df[df["ref"].str.match(r'^[ACGT]+$', na=False)]
    
    patho_df = df[df["clin_sig"].isin(pathogenic)].sample(10000, random_state=42)
    benign_df = df[df["clin_sig"].isin(benign)].sample(10000, random_state=42)
    
    sampled = pd.concat([patho_df, benign_df], ignore_index=True)
    
    label_map = {**{s: 1 for s in pathogenic}, **{s: 0 for s in benign}}
    sampled["label"] = sampled["clin_sig"].map(label_map)
    
    logging.info(f"Sampled {len(sampled):,} variants — {sampled['label'].sum():,} pathogenic, {(sampled['label']==0).sum():,} benign")
    
    annotated = annotate_variants(sampled)
    
    annotated = annotated.merge(
        sampled[["chrom", "pos", "label", "clin_sig", "cldn"]],
        on=["chrom", "pos"],
        how="left"
    )
    
    annotated.to_parquet(ANNOTATED_PATH, index=False)
    print(annotated.shape)
    print(annotated["label"].value_counts())
    print(annotated["consequence"].value_counts())