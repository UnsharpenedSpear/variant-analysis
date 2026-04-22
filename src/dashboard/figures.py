"""
Dashboard figures module - Data loading and chart generation.
Loads data once at module level to avoid re-reading files on every callback.
"""

from plotly import data
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, accuracy_score, f1_score, roc_auc_score
)

# ============================================================================
# MODULE 1: Imports and data loading
# ============================================================================

# Resolve project root regardless of where script is run from
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATURES_PATH = os.path.join(ROOT, "data", "processed", "features.parquet")
TEST_DATA_PATH = os.path.join(ROOT, "data", "processed", "test_data.parquet")
MODEL_PATH = os.path.join(ROOT, "results", "model.joblib")
SELECTED_FEATURES_PATH = os.path.join(ROOT, "results", "selected_features.json")
ANNOTATED_PATH = os.path.join(ROOT, "data", "processed", "clinvar_annotated.parquet")


def load_all_data():
    """
    Load all required data files once at module level.
    
    Returns:
        dict: Contains keys 'features', 'test', 'model', 'annotated', 'selected_features'
    """
    data = {}
    try:
        data['features'] = pd.read_parquet(FEATURES_PATH)
    except FileNotFoundError:
        data['features'] = pd.DataFrame()
    
    try:
        data['test'] = pd.read_parquet(TEST_DATA_PATH)
    except FileNotFoundError:
        data['test'] = pd.DataFrame()
    
    try:
        data['model'] = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        data['model'] = None
    
    try:
        data['annotated'] = pd.read_parquet(ANNOTATED_PATH)
    except FileNotFoundError:
        data['annotated'] = pd.DataFrame()
    
    try:
        with open(SELECTED_FEATURES_PATH, 'r') as f:
            feature_info = json.load(f)
            # extract just the names list from the dict
            if isinstance(feature_info, dict):
                data['selected_features'] = feature_info.get("names", [])
            else:
                data['selected_features'] = feature_info
    except FileNotFoundError:
        data['selected_features'] = []
    
    return data


# Load data at module level - singleton pattern
_cached_data = load_all_data()


# ============================================================================
# MODULE 2: Consequence distribution figure
# ============================================================================

def fig_consequence_distribution(df):
    """
    Plot distribution of genetic consequences.
    
    Args:
        df: DataFrame with 'consequence' column (from annotated data)
    
    Returns:
        plotly.graph_objects.Figure: Bar chart of consequence counts
    """
    if df.empty or 'consequence' not in df.columns:
        return px.bar(title="Consequence Distribution - No Data Available")
    
    consequence_counts = df['consequence'].value_counts().sort_values(ascending=False)
    
    fig = px.bar(
        x=consequence_counts.index,
        y=consequence_counts.values,
        labels={'x': 'Consequence', 'y': 'Count'},
        title='Genetic Consequence Distribution',
        color=consequence_counts.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig


# ============================================================================
# MODULE 3: Chromosome distribution figure
# ============================================================================

def fig_chrom_distribution(df):
    """
    Plot distribution of variants across chromosomes in biological order.
    
    Args:
        df: DataFrame with 'chrom' column
    
    Returns:
        plotly.graph_objects.Figure: Bar chart of variants per chromosome
    """
    if df.empty or 'chrom' not in df.columns:
        return px.bar(title="Chromosome Distribution - No Data Available")
    
    chrom_counts = df['chrom'].value_counts()
    
    # Biological chromosome order
    chrom_order = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
    chrom_counts = chrom_counts.reindex(
        [c for c in chrom_order if c in chrom_counts.index],
        fill_value=0
    )
    
    fig = px.bar(
        x=chrom_counts.index,
        y=chrom_counts.values,
        labels={'x': 'Chromosome', 'y': 'Variant Count'},
        title='Variant Distribution by Chromosome',
        color=chrom_counts.values,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig


# ============================================================================
# MODULE 4: ROC curve figure
# ============================================================================

def fig_roc_curve(model, X_test, y_test):
    """
    Plot ROC curve with AUC score and random baseline.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
    
    Returns:
        plotly.graph_objects.Figure: ROC curve figure
    """
    if model is None or X_test.empty or y_test.empty:
        return go.Figure().add_annotation(text="ROC Curve - No Model Available")
    
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Random baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='#d62728', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title=f'ROC Curve (AUC = {roc_auc:.3f})',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        hovermode='closest',
        height=500,
        template='plotly_white',
        xaxis=dict(scaleanchor='y', scaleratio=1),
        yaxis=dict(scaleanchor='x', scaleratio=1)
    )
    
    return fig


# ============================================================================
# MODULE 5: Precision-recall curve figure
# ============================================================================

def fig_precision_recall(model, X_test, y_test):
    """
    Plot precision-recall curve with no-skill baseline.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
    
    Returns:
        plotly.graph_objects.Figure: Precision-recall curve figure
    """
    if model is None or X_test.empty or y_test.empty:
        return go.Figure().add_annotation(text="P-R Curve - No Model Available")
    
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    
    # No-skill baseline = proportion of positive samples
    no_skill = y_test.mean()
    
    fig = go.Figure()
    
    # Precision-recall curve
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name='Precision-Recall Curve',
        line=dict(color='#2ca02c', width=2)
    ))
    
    # No-skill baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[no_skill, no_skill],
        mode='lines',
        name=f'No-Skill Baseline ({no_skill:.3f})',
        line=dict(color='#ff7f0e', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        hovermode='closest',
        height=500,
        template='plotly_white',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    return fig


# ============================================================================
# MODULE 6: Confusion matrix figure
# ============================================================================

def fig_confusion_matrix(model, X_test, y_test):
    """
    Plot confusion matrix heatmap.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
    
    Returns:
        plotly.graph_objects.Figure: Confusion matrix heatmap
    """
    if model is None or X_test.empty or y_test.empty:
        return go.Figure().add_annotation(text="Confusion Matrix - No Model Available")
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Benign", "Pathogenic"],
        y=["Benign", "Pathogenic"],
        text_auto=True,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        title='Confusion Matrix',
        height=400,
        width=500,
        template='plotly_white'
    )
    
    return fig


# ============================================================================
# MODULE 7: Feature importance figure
# ============================================================================

def fig_feature_importance(model, feature_names):
    """
    Plot top 20 most important features from Random Forest component.
    
    Args:
        model: Trained VotingClassifier with 'rf' estimator
        feature_names: List of feature names
    
    Returns:
        plotly.graph_objects.Figure: Horizontal bar chart of feature importance
    """
    if model is None or not feature_names:
        return px.bar(title="Feature Importance - No Model Available")
    
    # Access Random Forest from VotingClassifier
    try:
        rf_model = model.named_estimators_['rf']
        importances = rf_model.feature_importances_
    except (AttributeError, KeyError):
        # Fallback if structure is different
        return px.bar(title="Feature Importance - Model Structure Mismatch")
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True).tail(20)
    
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 20 Most Important Features',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        height=500,
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig


# ============================================================================
# MODULE 8: Metric summary cards data
# ============================================================================

def get_summary_metrics(model, X_test, y_test):
    """
    Calculate summary metrics for performance cards.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
    
    Returns:
        dict: Dictionary with keys accuracy, f1, roc_auc, and data counts
    """
    if model is None or X_test.empty or y_test.empty:
        return {
            'accuracy': 0.0,
            'f1': 0.0,
            'roc_auc': 0.0,
            'total_variants': 0,
            'pathogenic_count': 0,
            'benign_count': 0
        }
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'total_variants': len(y_test),
        'pathogenic_count': int((y_test == 1).sum()),
        'benign_count': int((y_test == 0).sum())
    }
    
    return metrics
        