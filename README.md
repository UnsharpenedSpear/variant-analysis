# Variant Analysis Portfolio

A bioinformatics ML pipeline for classifying genetic variants as pathogenic or benign using ClinVar data and Ensembl VEP annotations.

## Overview

This project implements an end-to-end variant pathogenicity classification pipeline:

- **Data source**: ClinVar database (GRCh37), ~16,000 annotated variants
- **Annotation**: Ensembl VEP REST API (consequence types, gene impact, novelty)
- **Features**: 31 engineered genomic features (nucleotide chemistry, consequence encoding, chromosomal position)
- **Model**: Ensemble (RandomForest + XGBoost) with SMOTE balancing and threshold tuning
- **Dashboard**: Interactive Plotly/Dash app for exploring predictions and model performance

## Performance

| Metric | Score |
|--------|-------|
| Accuracy | 60.98% |
| Precision | 51.76% |
| Recall | 77.12% |
| F1-Score | 61.94% |
| ROC AUC | 69.60% |
| Optimal threshold | 0.35 |

High recall (77%) is intentional — in clinical genomics, missing a pathogenic variant is worse than a false positive. The threshold is tuned to prioritise sensitivity over specificity.

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|-------|----------|-----------|--------|----|---------|
| Random Forest | 64.87% | 57.94% | 53.58% | 55.67% | 68.39% |
| XGBoost | 64.78% | 56.68% | 61.40% | 58.94% | 69.87% |
| Logistic Regression | 41.17% | 51.76% | 77.12% | 61.94% | 69.60% |
| **Ensemble** | **60.98%** | **51.76%** | **77.12%** | **61.94%** | **69.60%** |

## Top 10 Features

1. `pos` — genomic position
2. `chrom_encoded` — chromosome as integer
3. `is_transition` — Ti/Tv substitution type
4. `alt_is_gc` — alternate allele GC content
5. `ref_is_gc` — reference allele GC content
6. `is_novel` — variant novelty (no known rsID)
7. `csq_intron_variant` — intronic consequence
8. `csq_intergenic_variant` — intergenic consequence
9. `csq_downstream_gene_variant` — downstream region
10. `csq_upstream_gene_variant` — upstream region

## Pipeline Architecture

```
ClinVar VCF
    → parse & label (pathogenic=1, benign=0)
    → sample 10k pathogenic + 10k benign
    → VEP annotation (consequence, impact, gene)
    → feature engineering (31 features)
    → SMOTE balancing
    → ensemble training
    → threshold-tuned evaluation
    → Dash dashboard
```

## Project Structure

```
variant-analysis-portfolio/
├── data/
│   ├── raw/                        # Downloaded VCF files
│   ├── processed/                  # Parsed and engineered data
│   │   ├── clinvar.parquet         # Parsed ClinVar variants
│   │   ├── clinvar_annotated.parquet  # VEP-annotated with labels
│   │   ├── features.parquet        # Final ML feature matrix
│   │   ├── feature_names.json      # Feature column names
│   │   ├── train_data.parquet      # Training split (80%)
│   │   └── test_data.parquet       # Test split (20%)
│   └── truth/
│       ├── clinvar.vcf.gz          # Raw ClinVar VCF (GRCh37)
│       └── clinvar.parquet         # Parsed ClinVar labels
├── results/
│   ├── model.joblib                # Trained ensemble model
│   ├── selected_features.json      # Top 20 features used
│   └── evaluation_metrics_*.json   # Timestamped evaluation results
├── src/
│   ├── ingestion/
│   │   ├── clinvar.py              # ClinVar download and parsing
│   │   ├── download.py             # 1000 Genomes download
│   │   └── parse_vcf.py            # VCF parsing utility
│   ├── annotation/
│   │   └── vep_api.py              # Ensembl VEP REST API client
│   ├── analysis/
│   │   └── metrics.py              # Ti/Tv ratio, variant QC metrics
│   ├── ml/
│   │   ├── features.py             # Feature engineering pipeline
│   │   ├── train.py                # Model training with GridSearchCV
│   │   └── evaluate.py             # Threshold-tuned evaluation
│   └── dashboard/
│       ├── app.py                  # Dash app entry point
│       ├── layout.py               # Page structure and tab components
│       ├── figures.py              # Plotly figure generation
│       └── callbacks.py            # Interactive filter callbacks
├── tests/
│   ├── test_parse_vcf.py
│   ├── test_metrics.py
│   └── test_features.py
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_annotation.ipynb
│   └── 03_ml_model.ipynb
├── requirements.txt
├── Makefile
├── METHODS.md
└── README.md
```

## Setup

```bash
# Clone and navigate to project
cd variant-analysis-portfolio

# Install dependencies
pip install -r requirements.txt
```

## Running the Pipeline

```bash
# 1. Download and parse ClinVar
python src/ingestion/clinvar.py

# 2. Annotate with VEP (requires internet, takes ~20 mins)
python src/annotation/vep_api.py

# 3. Engineer features
python src/ml/features.py

# 4. Train model
python src/ml/train.py

# 5. Evaluate
python src/ml/evaluate.py
```

## Running the Dashboard

```bash
# From project root
python src/dashboard/app.py

# Visit http://127.0.0.1:8050
```

### Dashboard Tabs

1. **Dataset Overview** — variant counts, class balance, consequence and chromosome distribution
2. **Model Performance** — ROC curve, precision-recall curve, confusion matrix, metric cards
3. **Feature Importance** — top 20 features from the Random Forest component
4. **Variant Explorer** — filterable table of test set predictions by consequence, predicted label, and actual label

## Changing the Model

Edit `MODEL_TYPE` in `src/ml/train.py`:

```python
MODEL_TYPE = 'ensemble'  # 'rf', 'xgb', 'lr', or 'ensemble'
```

## Key Design Decisions

- **ClinVar as ground truth**: Labels come from expert-curated clinical submissions rather than proxy metrics, making the classifier scientifically defensible
- **Balanced sampling**: 10k pathogenic and 10k benign variants sampled before annotation to avoid class imbalance at the source
- **SMOTE**: Additional synthetic oversampling applied during training to correct residual imbalance
- **Feature selection**: Top 20 features by importance reduces noise and overfitting without significant performance loss
- **Threshold tuning**: Optimal threshold found by maximising F1 across the 0.3–0.8 range rather than defaulting to 0.5

## Future Work

- Add CADD scores as continuous pathogenicity features (regression extension)
- UMAP clustering of variant landscape as unsupervised analysis tab
- Deploy as public REST API via FastAPI + Render
- Expand to multi-class prediction (Pathogenic / Likely Pathogenic / VUS / Benign)
- Add cross-validation stability metrics and confidence intervals

## References

- ClinVar: https://www.ncbi.nlm.nih.gov/clinvar/
- Ensembl VEP: https://www.ensembl.org/info/docs/tools/vep/index.html
- 1000 Genomes Project: https://www.internationalgenome.org/

## Author

UnsharpenedSpear — April 2026

## License

MIT
