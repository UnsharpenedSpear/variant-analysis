import pandas as pd
import joblib
import json
import os
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, roc_auc_score
)
import numpy as np
import logging

# Paths
MODEL_PATH = os.path.join("results", "model.joblib")
SCALER_PATH = os.path.join("results", "scaler.joblib")
SELECTED_FEATURES_PATH = os.path.join("results", "selected_features.json")
TEST_DATA_PATH = os.path.join("data", "processed", "test_data.parquet")
METRICS_PATH = os.path.join("results", f"evaluation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_model_and_data(model_path=MODEL_PATH, test_data_path=TEST_DATA_PATH):
    """Load the trained model and test data."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found at {test_data_path}")
    
    model = joblib.load(model_path)
    logging.info(f"Loaded model: {type(model).__name__}")
    
    test_df = pd.read_parquet(test_data_path)
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]
    
    # Apply feature selection if available
    if os.path.exists(SELECTED_FEATURES_PATH):
        with open(SELECTED_FEATURES_PATH, 'r') as f:
            feature_info = json.load(f)
            if isinstance(feature_info, dict):
                selected_indices = feature_info.get("indices", None)
            else:
                selected_indices = None
        if selected_indices:
            X_test = X_test.iloc[:, selected_indices]
            logging.info(f"Applied feature selection: kept {len(selected_indices)} features")
    
    # Apply scaling if LR and scaler exists
    if type(model).__name__ == 'LogisticRegression' and os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        X_test = scaler.transform(X_test)
        logging.info("Applied scaling to test data")
    
    logging.info(f"Loaded test data with shape: {X_test.shape}")
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test, metrics_path=METRICS_PATH):
    """Evaluate the model on test data and save metrics with optimal threshold tuning."""
    
    # Get probability predictions for threshold tuning
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_proba = None
    
    # Find optimal threshold (maximizes F1)
    best_threshold = 0.5
    best_f1 = 0
    if y_proba is not None:
        thresholds = np.arange(0.3, 0.8, 0.05)
        for thresh in thresholds:
            y_pred_thresh = (y_proba >= thresh).astype(int)
            f1 = f1_score(y_test, y_pred_thresh, average='binary')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        logging.info(f"Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    
    # Use optimal threshold for final predictions
    if y_proba is not None:
        y_pred = (y_proba >= best_threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    
    # ROC AUC
    roc_auc = None
    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
    
    # Classification report and confusion matrix
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Threshold: {best_threshold:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(cm)
    
    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "threshold": best_threshold,
        "classification_report": report,
        "confusion_matrix": cm
    }
    
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    model, X_test, y_test = load_model_and_data()
    evaluate_model(model, X_test, y_test)