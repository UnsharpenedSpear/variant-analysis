import pandas as pd
import os
import json
import numpy as np
import logging

ANNOTATED_PATH = os.path.join("data", "processed", "clinvar_annotated.parquet")
FEATURES_PATH = os.path.join("data", "processed", "features.parquet")
FEATURE_NAMES_PATH = os.path.join("data", "processed", "feature_names.json")

def load_and_clean(path=ANNOTATED_PATH):
    df = pd.read_parquet(path)
    logging.info(f"Loaded {len(df)} variants from {path}")
    logging.info(f"Shape: {df.shape}")

    df = df.drop_duplicates(subset=["chrom", "pos", "ref", "alt"])
    df = df.dropna(subset=["label"])
    df.label = df.label.astype(int) 

    logging.info(f"After dropping duplicates and missing labels: {len(df)} variants")
    logging.info(f"Shape: {df.shape}")
    return df  

def  encode_chrom(df : pd.DataFrame) -> pd.DataFrame:
    chrom_map = {str(i): i for i in range(1, 23)}
    chrom_map.update({"X": 23, "Y": 24, "MT": 25})
    df["chrom_encoded"] = df["chrom"].map(chrom_map).fillna(-1).astype(int)
    return df

def add_nucleotide_features(df : pd.DataFrame) -> pd.DataFrame:
    transitions = {"AG", "GA", "CT", "TC"}
    df["is_transition"] = (df["ref"] + df["alt"]).isin(transitions).astype(int)
    df["ref_is_gc"] = df["ref"].isin(["G", "C"]).astype(int)
    df["alt_is_gc"] = df["alt"].isin(["G", "C"]).astype(int)    
    return df

def add_variant_features(df : pd.DataFrame) -> pd.DataFrame:
    df["is_novel"] = (df["rs_id"] == "novel").astype(int)
    coding_consequences = {"missense_variant", "synonymous_variant", "stop_gained", "stop_lost", "start_lost", "frameshift_variant", "inframe_insertion", "inframe_deletion", "splice_donor_variant", "splice_acceptor_variant", "protein_altering_variant"}
    df["is_coding"] = df["consequence"].isin(coding_consequences).astype(int)
    return df

def encode_consequence(df : pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(df["consequence"], prefix="csq", dtype=int)

    df = pd.concat([df, dummies], axis=1)
    df.drop(columns=["consequence"], inplace=True)
    logging.info(f"Encoded consequence into {dummies.shape[1]} features")
    return df

def finalise_features(df : pd.DataFrame) -> pd.DataFrame:
    df["is_novel"] = df["is_novel"].astype(int)
    df["is_coding"] = df["is_coding"].astype(int)
    df.drop(columns=["gene_id", "gene_symbol","rs_id", "clin_sig", "cldn", "impact", "chrom", "ref", "alt"], inplace=True, errors="ignore")
    df.fillna(0, inplace=True)
    return df   

def save_features(df : pd.DataFrame, features_path=FEATURES_PATH, feature_names_path=FEATURE_NAMES_PATH):
    label = df["label"]
    features = df.drop(columns=["label"])
    feature_names = features.columns.tolist()
    df = pd.concat([features, label], axis=1)
    df.to_parquet(features_path, index=False)
    logging.info(f"Saved features to {features_path}")
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    logging.info(f"Saved feature names to {feature_names_path}")    
    logging.info(f"Final feature set has {len(feature_names)} features")


if __name__ == "__main__":
    df = load_and_clean(ANNOTATED_PATH)
    df = encode_chrom(df)
    df = add_nucleotide_features(df)
    df = add_variant_features(df)
    df = encode_consequence(df)
    df = finalise_features(df)
    save_features(df)
    print(df.shape)
    print(df.dtypes)
    print(df["label"].value_counts())