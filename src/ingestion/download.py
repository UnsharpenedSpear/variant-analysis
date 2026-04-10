import requests
import os   

DOWNLOAD_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/pilot_data/release/2010_07/"
    "trio/snps/CEU.trio.2010_03.genotypes.vcf.gz"
)

OUTPUT_PATH = os.path.join("data", "raw", "variants.vcf.gz")


def download_vcf(url: str = DOWNLOAD_URL, output_path: str = OUTPUT_PATH) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"File already exists at {output_path}, skipping download.")
        return output_path

    print(f"Downloading VCF from 1000 Genomes...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:.1f}% ({downloaded // 1_000_000}MB / {total // 1_000_000}MB)", end="")

    print(f"\nDone. Saved to {output_path}")
    return output_path


if __name__ == "__main__":
    download_vcf()