"""
Dashboard layout module - UI components and tab layouts.
Assembles all figures and creates a responsive multi-tab interface.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
from figures import (
    load_all_data,
    fig_consequence_distribution,
    fig_chrom_distribution,
    fig_roc_curve,
    fig_precision_recall,
    fig_confusion_matrix,
    fig_feature_importance,
    get_summary_metrics
)

# ============================================================================
# Load data at module level for use in layouts
# ============================================================================

data = load_all_data()


# ============================================================================
# MODULE 2: Summary card component
# ============================================================================

def make_card(title, value, color="primary"):
    """
    Create a reusable metric card for displaying summary statistics.
    
    Args:
        title (str): Card title label
        value: Numeric or string value to display
        color (str): Bootstrap color - "primary", "success", "danger", "warning"
    
    Returns:
        dbc.Card: A styled card component
    """
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted small mb-2"),
            html.H4(f"{value}", className="fw-bold")
        ]),
        color=color,
        outline=True,
        className="text-center"
    )


# ============================================================================
# MODULE 3: Tab 1 layout (dataset overview)
# ============================================================================

def tab_dataset():
    """
    Dataset overview tab showing data distribution and composition.
    
    Returns:
        html.Div: Tab content
    """
    # Calculate summary metrics from test data
    if not data['test'].empty:
        total_variants = len(data['test'])
        pathogenic_count = int((data['test'].get('label', pd.Series()) == 1).sum())
        benign_count = total_variants - pathogenic_count
        ratio = f"{pathogenic_count}:{benign_count}" if total_variants > 0 else "N/A"
    else:
        total_variants = 0
        pathogenic_count = 0
        benign_count = 0
        ratio = "N/A"
    
    return html.Div([
        dbc.Row([
            dbc.Col(make_card("Total Variants", total_variants, "primary"), md=3),
            dbc.Col(make_card("Pathogenic", pathogenic_count, "danger"), md=3),
            dbc.Col(make_card("Benign", benign_count, "success"), md=3),
            dbc.Col(make_card("Class Ratio P:B", ratio, "warning"), md=3),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_consequence_distribution(data['annotated'])), md=6),
            dbc.Col(dcc.Graph(figure=fig_chrom_distribution(data['annotated'])), md=6),
        ])
    ], className="p-3")


# ============================================================================
# MODULE 4: Tab 2 layout (model performance)
# ============================================================================

def tab_model():
    """
    Model performance tab showing evaluation metrics and diagnostic plots.
    
    Returns:
        html.Div: Tab content
    """
    # Calculate metrics
    if data['model'] is not None and not data['test'].empty:
        metrics = get_summary_metrics(
            data['model'],
            data['test'][data['selected_features']] if data['selected_features'] else data['test'],
            data['test'].get('label', pd.Series())
        )
    else:
        metrics = {
            'accuracy': 0.0,
            'f1': 0.0,
            'roc_auc': 0.0
        }
    
    return html.Div([
        dbc.Row([
            dbc.Col(make_card("Accuracy", f"{metrics['accuracy']:.3f}", "primary"), md=3),
            dbc.Col(make_card("F1 Score", f"{metrics['f1']:.3f}", "success"), md=3),
            dbc.Col(make_card("ROC AUC", f"{metrics['roc_auc']:.3f}", "danger"), md=3),
            dbc.Col(make_card("Threshold", "0.5", "warning"), md=3),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col(
                dcc.Graph(figure=fig_roc_curve(
                    data['model'],
                    data['test'][data['selected_features']] if data['selected_features'] else data['test'],
                    data['test'].get('label', pd.Series())
                )),
                md=6
            ),
            dbc.Col(
                dcc.Graph(figure=fig_precision_recall(
                    data['model'],
                    data['test'][data['selected_features']] if data['selected_features'] else data['test'],
                    data['test'].get('label', pd.Series())
                )),
                md=6
            ),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col(
                dcc.Graph(figure=fig_confusion_matrix(
                    data['model'],
                    data['test'][data['selected_features']] if data['selected_features'] else data['test'],
                    data['test'].get('label', pd.Series())
                )),
                md=12
            ),
        ])
    ], className="p-3")


# ============================================================================
# MODULE 5: Tab 3 layout (feature importance)
# ============================================================================

def tab_features():
    """
    Feature importance tab showing which features drive model predictions.
    
    Returns:
        html.Div: Tab content
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.P(
                    "Feature importance is derived from the Random Forest component of the ensemble model. "
                    "Features with higher importance scores contribute more significantly to the model's predictions. "
                    "Understanding feature importance helps identify key genetic markers associated with pathogenicity.",
                    className="text-muted"
                )
            ], md=12)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col(
                dcc.Graph(figure=fig_feature_importance(
                    data['model'],
                    data['selected_features']
                )),
                md=12
            ),
        ])
    ], className="p-3")


# ============================================================================
# MODULE 6: Tab 4 layout (variant explorer)
# ============================================================================

def tab_explorer():
    """
    Variant explorer tab for filtering and viewing individual variants.
    
    Returns:
        html.Div: Tab content
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Filter by Consequence", className="fw-bold"),
                dcc.Dropdown(
                    id="filter-consequence",
                    options=[{"label": "All", "value": None}] + (
                        [{"label": c, "value": c} for c in sorted(data['annotated']['consequence'].unique())]
                        if not data['annotated'].empty and 'consequence' in data['annotated'].columns
                        else []
                    ),
                    value=None,
                    clearable=True
                )
            ], md=4),
            dbc.Col([
                html.Label("Filter by Predicted", className="fw-bold"),
                dcc.Dropdown(
                    id="filter-predicted",
                    options=[
                        {"label": "All", "value": None},
                        {"label": "Benign (0)", "value": 0},
                        {"label": "Pathogenic (1)", "value": 1}
                    ],
                    value=None,
                    clearable=True
                )
            ], md=4),
            dbc.Col([
                html.Label("Filter by Actual", className="fw-bold"),
                dcc.Dropdown(
                    id="filter-actual",
                    options=[
                        {"label": "All", "value": None},
                        {"label": "Benign (0)", "value": 0},
                        {"label": "Pathogenic (1)", "value": 1}
                    ],
                    value=None,
                    clearable=True
                )
            ], md=4),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                html.Div(id="variant-table-container")
            ], md=12)
        ])
    ], className="p-3")


# ============================================================================
# MODULE 7: Main layout
# ============================================================================

def create_layout():
    """
    Create the main page layout with tabbed interface.
    
    Returns:
        dbc.Container: Full page structure
    """
    return dbc.Container(
        [
            html.H1("Variant Pathogenicity Prediction Dashboard", className="mt-4 mb-4"),
            html.Hr(),
            
            dbc.Tabs([
                dbc.Tab(tab_dataset(), label="📊 Dataset Overview", tab_id="dataset"),
                dbc.Tab(tab_model(), label="🎯 Model Performance", tab_id="model"),
                dbc.Tab(tab_features(), label="⭐ Feature Importance", tab_id="features"),
                dbc.Tab(tab_explorer(), label="🔍 Variant Explorer", tab_id="explorer"),
            ], id="tabs", active_tab="dataset")
        ],
        fluid=True,
        className="py-4"
    )
