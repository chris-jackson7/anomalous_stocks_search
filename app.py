import pickle
import numpy as np
import pandas

import plotly.express as px
import dash
from dash import html, dcc
from dash.dependencies import Input, Output


# Load Source Data
with open('search_df.pkl', 'rb') as file:
    search_df = pickle.load(file)


# Define Initial Parameters
anomaly_metric = 'iso'
high = 0.2
low = 0.1

# Define Filter Function
def filter_data(data, metric, high_thresh, low_thresh):
    search_mask = np.array(data['industry'] != 'Biotechnology') & \
                  np.array(data['industry'] != 'Drug Manufacturers - Specialty & Generic') & \
                  np.array(data[metric] < high_thresh) & \
                  np.array(data[metric] > low_thresh)
    return data.iloc[search_mask] #.sort_values(metric, ascending=False)


# Initialize App
app = dash.Dash(__name__)

# App Layout
app.layout = html.Div([
    html.H1("Interactive Stock Anomalies"),
    html.Div(style={'textAlign': 'center'}, children=[
        html.H3("Anomaly Distributions", style={'marginBottom': 50}),
        html.Img(id="static-plot", src=app.get_asset_url("anomaly_distributions.png"), style={'width': '100%', 'max-width': '800px'}),
    ]),
    html.Div([
        html.Label("Anomaly Metric:"),
        dcc.Dropdown(
            id="metric-dropdown",
            options=[{'label': 'Isolation Score', 'value': 'iso'},
                     {'label': 'Z-Score', 'value': 'z_score'},
                     {'label': 'Squared Error', 'value': 'se'}],
            value=anomaly_metric
        ),
    ]),
    html.Div([
        html.Label("High Threshold:"),
        dcc.Input(id="high-threshold", type="number", value=high),
        html.Label("Low Threshold:"),
        dcc.Input(id="low-threshold", type="number", value=low)
    ]),
    dcc.Graph(id="scatter-plot")
])

# Style for all elements
app.layout.style = {
    'backgroundColor': '#f5f5f5',
    'margin': '20px',
    'fontFamily': 'sans-serif',
}

# Update Callback
@app.callback(
    Output("scatter-plot", "figure"),
    [Input("metric-dropdown", "value"),
     Input("high-threshold", "value"),
     Input("low-threshold", "value")]
)
def update_plot(metric, high_thresh, low_thresh):
    filtered_data = filter_data(search_df.copy(), metric, high_thresh, low_thresh)

    JITTER = .2
    filtered_data["y"] = filtered_data["y"] + JITTER * (2 * np.random.rand(len(filtered_data)) - 1)
    filtered_data["pred"] = filtered_data["pred"] + JITTER * (2 * np.random.rand(len(filtered_data)) - 1)

    fig = px.scatter(filtered_data, x="y", y="pred", text=filtered_data.index)
    
    # Add Reference Line
    fig.add_shape(
        type="line",
        x0=min(filtered_data["y"]) - 1,
        y0=min(filtered_data["y"]) - 1,
        x1=max(filtered_data["pred"]) + 1,
        y1=max(filtered_data["pred"]) + 1,
        line=dict(color="red", dash="dash")
    )

    # Add Grid
    fig.update_layout(xaxis_showgrid=True, yaxis_showgrid=True)
    fig.update_layout(width=1600, height=800)
    return fig

# Run App
if __name__ == '__main__':
    app.run_server(debug=True)
