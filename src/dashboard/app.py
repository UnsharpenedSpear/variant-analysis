"""
Dashboard application entry point.
Initializes and runs the Dash web application with Bootstrap styling.
"""

from dash import Dash
import dash_bootstrap_components as dbc
from layout import create_layout
import callbacks  # Import to register callbacks

# ============================================================================
# Initialize Dash app with Bootstrap theme
# ============================================================================

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

# Set page title
app.title = "Variant Pathogenicity Dashboard"

# Set layout
app.layout = create_layout()


# ============================================================================
# Main block - Run server
# ============================================================================

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
