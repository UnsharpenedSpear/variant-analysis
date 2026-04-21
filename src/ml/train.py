import pandas as pd
import os
import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import logging

# Paths
FEATURES_PATH = os.path.join("data", "processed", "features.parquet")
FEATURE_NAMES_PATH = os.path.join("data", "processed", "feature_names.json")
MODEL_PATH = os.path.join("results", "model.joblib")
SCALER_PATH = os.path.join("results", "scaler.joblib")
SELECTED_FEATURES_PATH = os.path.join("results", "selected_features.json")
TRAIN_DATA_PATH = os.path.join("data", "processed", "train_data.parquet")
TEST_DATA_PATH = os.path.join("data", "processed", "test_data.parquet")

# Model selection: 'rf' for RandomForest, 'xgb' for XGBoost, 'lr' for LogisticRegression, 'ensemble' for VotingClassifier
MODEL_TYPE = 'ensemble'
TOP_N_FEATURES = 20  # Feature selection: use top N features by importance (None = all features)  

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(features_path=FEATURES_PATH, feature_names_path=FEATURE_NAMES_PATH):
    df = pd.read_parquet(features_path)
    logging.info(f"Loaded features with shape: {df.shape}")
    
    with open(feature_names_path, "r") as f:
        feature_names = json.load(f)
    logging.info(f"Feature names: {feature_names}")
    
    X = df[feature_names]
    y = df["label"]
    return X, y

def data_split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test

def train_model(X, y):
    # Feature selection: keep top N features by importance
    selected_feature_indices = None
    if TOP_N_FEATURES and TOP_N_FEATURES < X.shape[1]:
        # Train a quick RF to get feature importances
        temp_rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        temp_rf.fit(X, y)
        importances = temp_rf.feature_importances_
        top_indices = np.argsort(importances)[-TOP_N_FEATURES:][::-1]
        selected_feature_indices = top_indices.tolist()
        selected_features = X.columns[top_indices].tolist()
        X = X.iloc[:, top_indices]
        with open(SELECTED_FEATURES_PATH, 'w') as f:
            json.dump({"indices": selected_feature_indices, "names": selected_features}, f, indent=2)
        logging.info(f"Selected top {TOP_N_FEATURES} features: {selected_features}")
    
    if MODEL_TYPE == 'rf':
        model = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [100, 200, 500], 'max_depth': [10, 20, None], 'class_weight': [None, 'balanced']}
    elif MODEL_TYPE == 'xgb':
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 6, 10], 'learning_rate': [0.1, 0.2], 'scale_pos_weight': [1, len(y[y==0])/len(y[y==1])]}
    elif MODEL_TYPE == 'lr':
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {'C': [0.01, 0.1, 1.0], 'class_weight': [None, 'balanced']}
    elif MODEL_TYPE == 'ensemble':
        rf = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
        xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
        model = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='soft', n_jobs=-1)
        param_grid = {}
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")
    
    # Apply SMOTE for imbalance with optimized sampling
    smote = SMOTE(random_state=42, sampling_strategy='auto')  # Balance to 50-50
    X_sm, y_sm = smote.fit_resample(X, y)
    logging.info(f"Applied SMOTE: original shape {X.shape}, resampled shape {X_sm.shape}")
    
    # Scaling for LR
    scaler = None
    if MODEL_TYPE == 'lr':
        scaler = StandardScaler()
        X_sm = scaler.fit_transform(X_sm)
        joblib.dump(scaler, SCALER_PATH)
        logging.info(f"Scaler saved to {SCALER_PATH}")
    
    if param_grid:
        grid = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid.fit(X_sm, y_sm)
        logging.info(f"Best params for {MODEL_TYPE}: {grid.best_params_}")
        logging.info(f"Best CV score: {grid.best_score_:.4f}")
        return grid.best_estimator_
    else:
        # For ensemble, fit directly
        model.fit(X_sm, y_sm)
        logging.info(f"Ensemble model trained successfully")
        return model

def save_splits(X_train, X_test, y_train, y_test):

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_parquet(TRAIN_DATA_PATH, index=False)
    test_df.to_parquet(TEST_DATA_PATH, index=False)
    
    logging.info(f"Saved training data to {TRAIN_DATA_PATH}")
    logging.info(f"Saved test data to {TEST_DATA_PATH}")

def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logging.info(f"Model saved to {path}")

if __name__ == "__main__":
    X, y = load_data()
    
    X_train, X_test, y_train, y_test = data_split(X, y)

    model = train_model(X_train, y_train)
    
    # Save the model
    save_model(model)
    
    # Save the splits for evaluation
    save_splits(X_train, X_test, y_train, y_test)