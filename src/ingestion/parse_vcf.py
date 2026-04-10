import vcf
import pandas as pd
import os
import gzip
import shutil

RAW_PATH_GZ = os.path.join("data", "raw", "variants.vcf.gz")
PROCESSED_PATH = os.path.join("data", "processed", "variants.parquet")

def parse_vcf(input_gz: str = RAW_PATH_GZ, output_path: str = PROCESSED_PATH) -> pd.DataFrame:
    uncompressed_path = input_gz.replace('.gz', '')
    
    # 1. BYPASS: Manually extract the .vcf.gz to a standard .vcf file first
    print("Decompressing file to bypass PyVCF3 bugs...")
    if not os.path.exists(uncompressed_path):
        try:
            with gzip.open(input_gz, 'rb') as f_in:
                with open(uncompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except gzip.BadGzipFile:
            # If it wasn't actually gzipped, just copy it over
            shutil.copy2(input_gz, uncompressed_path)

    print(f"Parsing raw text VCF: {uncompressed_path}")
    
    # 2. Parse the plain text file using standard open()
    records = []
    with open(uncompressed_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = vcf.Reader(f)
        for record in reader:
            alt_val = str(record.ALT[0]) if record.ALT and record.ALT[0] else None
            
            records.append({
                "chrom": record.CHROM,
                "pos": record.POS,
                "ref": record.REF,
                "alt": alt_val,
                "qual": record.QUAL,
                "filter": str(record.FILTER[0]) if record.FILTER else "PASS",
                "dp": record.INFO.get("DP", None),
                "af": record.INFO.get("AF", [None])[0] if "AF" in record.INFO else None,
                "var_type": record.var_type,
            })

    if not records:
        print("Warning: No variants found.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"Successfully parsed {len(df):,} variants → saved to {output_path}")
    return df

if __name__ == "__main__":
    df = parse_vcf()
    if not df.empty:
        print("\n--- Data Preview ---")
        print(df.head())