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