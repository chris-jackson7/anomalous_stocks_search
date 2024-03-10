import pickle
import numpy as np
import pandas

import plotly.express as px
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output

from scripts.constants import TARGET_COLUMN


# Load Source Data
with open('search_df.pkl', 'rb') as file:
    search_df = pickle.load(file)

with open('assets/model_rmse.txt', 'r') as file:
    RMSE = file.read()

# search_df['color'] = [0] * len(search_df)

# Define Initial Parameters
anomaly_metric = 'iso'
low = 0.1
high = 0.2
sector = ''
remove_drug_makers = False
# user_input = ''

# Define Filter Function
def filter_data(data, metric, high_thresh, low_thresh, sector, remove_drug_makers):
    search_mask = np.array(data[metric] < high_thresh) & \
                  np.array(data[metric] > low_thresh)
    if sector:
        search_mask = search_mask & np.array(data['sector'] == sector)
    if remove_drug_makers:
        search_mask = search_mask & \
                  np.array(data['industry'] != 'Biotechnology') & \
                  np.array(data['industry'] != 'Drug Manufacturers - Specialty & Generic')
    print(sum(search_mask))
    return data.iloc[search_mask].sort_values(metric, ascending=False)


# Initialize App
app = dash.Dash(__name__)

server = app.server

# App Layout
app.layout = html.Div([
    html.H1("Interactive Stock Anomalies"),
    html.Div(style={'textAlign': 'center', 'marginBottom': 50}, children=[
        html.H3("XG Boost Model Fits & Feature Importance", style={'marginBottom': 25}),
        html.P(f"target column: {TARGET_COLUMN} | model RMSE: {RMSE}", style={'marginBottom': 25}),
        html.Img(id="model-fit-plot", src=app.get_asset_url("train_test_fit_feature_importance.png"), style={'width': '100%', 'max-width': '1600px'}),
    ]),
    html.Div(style={'textAlign': 'center', 'marginBottom': 50}, children=[
        html.H3("Anomaly Distributions", style={'marginBottom': 50}),
        html.Img(id="static-plot", src=app.get_asset_url("anomaly_distributions.png"), style={'width': '100%', 'max-width': '1600px'}),
    ]),
    html.Div(
        style={
            'display': 'flex',
            'flex-direction': 'column',  # Stack vertically
            'align-items': 'center',  # Center horizontally
            'width': '100%'  # Take full width
        },
        children=[
            html.Div(style={
                        'display': 'flex',
                        'justify-content': 'center',  # Center inputs horizontally
                        'width': '50%',  # Adjust width for inputs (optional)
                        'marginBottom': 50,
                        'margin-left': '10px',
                        'margin-right': '10px'
                    }, children=[
                        html.Div(style={
                                #'display': 'flex',
                                'justify-content': 'center',
                                'width': '50%',
                                'marginBottom': 25,
                            }, children = [
                                html.Label("Anomaly Metric:"),
                                dcc.Dropdown(
                                    id="metric-dropdown",
                                    options=[{'label': 'Isolation Score', 'value': 'iso'},
                                            {'label': 'Z-Score', 'value': 'z_score'},
                                            {'label': 'Squared Error', 'value': 'se'}],
                                    value=anomaly_metric
                                ),
                                html.Label("Sector:"),
                                dcc.Dropdown(
                                    id="sector-dropdown",
                                    options=[{'label': sector, 'value': sector} for sector in search_df['sector'].unique()],
                                    value=''
                                ),
                            ]),
                        html.Label("Low Threshold:"),
                        dcc.Input(id="low-threshold", type="number", value=low),  
                        html.Label("High Threshold:"),
                        dcc.Input(id="high-threshold", type="number", value=high),
                        html.Label("Drug Makers:"),
                        dcc.Checklist(
                            id="remove-drug-makers",
                            options=[
                                {'label': 'remove', 'value': True},
                            ],
                            value=remove_drug_makers
                        ),
            ]),
            dcc.Graph(id="scatter-plot")
    ]),
    html.Div([
        # html.Label('Search by Ticker(s) ex: APPL,MSFT,NVDA:'),
        # dcc.Input(id='user-input', type='text', value="", n_submit=0),
        html.Div(id='user-input-table')
    ])
])

# Style for all elements
app.layout.style = {
    'backgroundColor': '#f5f5f5',
    'margin': '20px',
    'fontFamily': 'sans-serif',
}

# Plot Update Callback
@app.callback(
    [Output("scatter-plot", "figure"),
     Output("user-input-table", "children")],
    [Input("metric-dropdown", "value"),
     Input("high-threshold", "value"),
     Input("low-threshold", "value"),
     Input("sector", "value"),
     Input("remove-drug-makers", "value")]
)
def update_plot(metric, high_thresh, low_thresh, sector, remove_drug_makers):
    try:
        filtered_data = filter_data(search_df.copy(), metric, high_thresh, low_thresh, sector, remove_drug_makers)

        # TODO: consider toggle for jittering
        # JITTER = .2
        # filtered_data["y"] = filtered_data["y"] + JITTER * (2 * np.random.rand(len(filtered_data)) - 1)
        # filtered_data["pred"] = filtered_data["pred"] + JITTER * (2 * np.random.rand(len(filtered_data)) - 1)

        fig = px.scatter(filtered_data, x="y", y="pred",
                        text=filtered_data.index
                        # color="color",
                        # color_discrete_sequence=px.colors.qualitative.Set3
                        )
        
        # Add Reference Line
        if not filtered_data.empty:
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

        return fig, search_index(','.join(filtered_data.index))
    except Exception as e:
        print(e)
    

# @app.callback(
#     Output("user-input-table", "children"),
#     Input("user-input", "n_submit"),
#     debounce_time=1000
# )
def search_index(user_input):
    if user_input != "" and user_input != "None":
        valid_user_input = [ticker for ticker in str(user_input).split(',') if ticker in search_df.index]
        if valid_user_input:
            search_df.loc[valid_user_input, 'color'] = 1
            
            df = search_df.copy()
            df = df.loc[valid_user_input, ['y', 'pred', 'se', 'iso', 'z_score', 'sector', 'industry']]
            df.insert(0, 'symbol', df.index)
            if not df.empty:
                table = dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in df.columns]
                )
                return table
        else:
            print('Input not in index')
            pass


# Run App
if __name__ == '__main__':
    app.run_server(debug=True)
