"""
Dashboard callbacks module - Reactive data filtering and updates.
Handles all user interactions and dynamic content updates.
"""

from dash import Input, Output, callback, html
import dash_bootstrap_components as dbc
import pandas as pd
import json
from figures import load_all_data

# ============================================================================
# MODULE 1: Imports and data loading
# ============================================================================

# Load data at module level - singleton pattern
data = load_all_data()


# ============================================================================
# MODULE 2: Variant explorer callback
# ============================================================================

@callback(
    Output("variant-table-container", "children"),
    Input("filter-consequence", "value"),
    Input("filter-predicted", "value"),
    Input("filter-actual", "value"),
)
def update_variant_table(filter_consequence, filter_predicted, filter_actual):
    """
    Filter and display variants based on dropdown selections.
    
    Args:
        filter_consequence: Selected consequence filter value
        filter_predicted: Selected predicted label filter (0/1/None)
        filter_actual: Selected actual label filter (0/1/None)
    
    Returns:
        dbc.Table or html.Div: Styled table or message
    """
    # Start with test data
    if data['test'].empty:
        return dbc.Alert("No test data available", color="warning")
    
    df = data['test'].copy()
    
    # Add predictions if model exists
    if data['model'] is not None and not df.empty:
        feature_cols = data['selected_features'] if data['selected_features'] else df.columns.tolist()
        # Only use columns that exist in the dataframe
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        if feature_cols:
            try:
                df['predicted_label'] = data['model'].predict(df[feature_cols])
                df['predicted_proba'] = data['model'].predict_proba(df[feature_cols])[:, 1]
            except Exception as e:
                return dbc.Alert(f"Error making predictions: {str(e)}", color="danger")
        else:
            df['predicted_label'] = 0
            df['predicted_proba'] = 0.5
    else:
        df['predicted_label'] = 0
        df['predicted_proba'] = 0.5
    
    # Apply consequence filter
    if filter_consequence is not None and 'consequence' in df.columns:
        df = df[df['consequence'] == filter_consequence]
    
    # Apply predicted label filter
    if filter_predicted is not None and 'predicted_label' in df.columns:
        df = df[df['predicted_label'] == filter_predicted]
    
    # Apply actual label filter
    if filter_actual is not None and 'label' in df.columns:
        df = df[df['label'] == filter_actual]
    
    if df.empty:
        return dbc.Alert("No variants match the selected filters", color="info")
    
    # Format and display table
    display_df = df.head(100).copy()  # Limit to 100 rows for performance
    
    # Rename columns for display
    display_columns = []
    for col in display_df.columns:
        if col == 'predicted_label':
            display_columns.append('Predicted')
        elif col == 'predicted_proba':
            display_columns.append('Confidence')
        elif col == 'label':
            display_columns.append('Actual')
        else:
            display_columns.append(col.replace('_', ' ').title())
    
    display_df.columns = display_columns
    
    # Format numeric columns
    for col in display_df.columns:
        if 'Confidence' in col or 'proba' in col.lower():
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    
    table = dbc.Table.from_dataframe(
        display_df,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="table-sm"
    )
    
    return html.Div([
        html.P(f"Showing {len(df)} variants (limited to 100 rows)", className="text-muted small"),
        table
    ])
