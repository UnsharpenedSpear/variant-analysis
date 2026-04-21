import requests
import pandas as pd 
import os
import gzip
import shutil
import vcf  
import logging

CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh37/clinvar.vcf.gz"
CLINVAR_RAW = os.path.join("data", "truth", "clinvar.vcf.gz")
CLINVAR_VCF = os.path.join("data", "truth", "clinvar.vcf")
CLINVAR_PARSED = os.path.join("data", "processed", "clinvar.parquet")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def download_clinvar(url=CLINVAR_URL, output_path=CLINVAR_RAW):
    downloaded = 0
    if os.path.exists(output_path):
        logging.info(f"ClinVar file already exists at {output_path}. Skipping download.")
        return output_path
    logging.info(f"Downloading ClinVar data from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded % (10 * 1024 * 1024) < 8192:
                logging.info(f"Downloaded {downloaded / (1024 * 1024):.1f} MB...")
    logging.info(f"Download complete: {output_path} ({os.path.getsize(output_path) / (1024 * 1024):.2f} MB)")
    return output_path

def decompress_clinvar(input_gz=CLINVAR_RAW, output_vcf=CLINVAR_VCF):
    if os.path.exists(output_vcf):
        logging.info(f"Decompressed ClinVar VCF already exists at {output_vcf}. Skipping decompression.")
        return output_vcf
    logging.info(f"Decompressing {input_gz} to {output_vcf}...")
    with gzip.open(input_gz, 'rb') as f_in:
        with open(output_vcf, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    logging.info(f"Decompression complete: {output_vcf} ({os.path.getsize(output_vcf) / (1024 * 1024):.2f} MB)")
    return output_vcf

def parse_clinvar(input_vcf=CLINVAR_VCF, output_parquet=CLINVAR_PARSED):
    if os.path.exists(output_parquet):
        logging.info(f"Parsed ClinVar data already exists at {output_parquet}. Skipping parsing.")
        return output_parquet
    logging.info(f"Parsing ClinVar VCF: {input_vcf}...")
    records = []
    with open(input_vcf, 'r', encoding='utf-8', errors='ignore') as f:
        reader = vcf.Reader(f)
        for record in reader:   
            records.append({
                "chrom": record.CHROM,
                "pos": record.POS,
                "ref": record.REF,
                "alt": str(record.ALT[0]    ) if record.ALT else None,
                "clin_sig": record.INFO.get("CLNSIG", ["unknown"])[0],
                "cldn": record.INFO.get("CLNDN", ["unknown"])[0],
            })
    df = pd.DataFrame(records)
    df.to_parquet(output_parquet, index=False)
    logging.info(f"Parsing complete: {len(df):,} variants saved to {output_parquet}")
    return output_parquet
    
if __name__ == "__main__":
    download_clinvar()
    decompress_clinvar()
    parse_clinvar()

