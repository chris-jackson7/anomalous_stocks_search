import pickle
import numpy as np
import pandas

import plotly.express as px
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output


# Load Source Data
with open('search_df.pkl', 'rb') as file:
    search_df = pickle.load(file)

search_df['color'] = [0] * len(search_df)

# Define Initial Parameters
anomaly_metric = 'iso'
high = 0.2
low = 0.1
user_input = ''

# Define Filter Function
def filter_data(data, metric, high_thresh, low_thresh):
    search_mask = np.array(data['industry'] != 'Biotechnology') & \
                  np.array(data['industry'] != 'Drug Manufacturers - Specialty & Generic') & \
                  np.array(data[metric] < high_thresh) & \
                  np.array(data[metric] > low_thresh)
    return data.iloc[search_mask] #.sort_values(metric, ascending=False)


# Initialize App
app = dash.Dash(__name__)

server = app.server

# App Layout
app.layout = html.Div([
    html.H1("Interactive Stock Anomalies"),
    html.Div(style={'textAlign': 'center', 'marginBottom': 50}, children=[
        html.H3("Anomaly Distributions", style={'marginBottom': 50}),
        html.Img(id="static-plot", src=app.get_asset_url("anomaly_distributions.png"), style={'width': '100%', 'max-width': '800px'}),
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
                            ]),     
                        html.Label("High Threshold:"),
                        dcc.Input(id="high-threshold", type="number", value=high),
                        html.Label("Low Threshold:"),
                        dcc.Input(id="low-threshold", type="number", value=low),
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
     Input("low-threshold", "value")]
)
def update_plot(metric, high_thresh, low_thresh):
    filtered_data = filter_data(search_df.copy(), metric, high_thresh, low_thresh)

    JITTER = .2
    filtered_data["y"] = filtered_data["y"] + JITTER * (2 * np.random.rand(len(filtered_data)) - 1)
    filtered_data["pred"] = filtered_data["pred"] + JITTER * (2 * np.random.rand(len(filtered_data)) - 1)

    fig = px.scatter(filtered_data, x="y", y="pred",
                    text=filtered_data.index, color="color",
                    color_discrete_sequence=px.colors.qualitative.Set3)
    
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

    return fig, search_index(','.join(filtered_data.index), metric)
    

# @app.callback(
#     Output("user-input-table", "children"),
#     Input("user-input", "n_submit"),
#     debounce_time=1000
# )
def search_index(user_input, metric):
    if user_input != "" and user_input != "None":
        valid_user_input = [ticker for ticker in str(user_input).split(',') if ticker in search_df.index]
        if valid_user_input:
            print(valid_user_input)
            search_df.loc[valid_user_input, 'color'] = 1
            
            df = search_df.copy()
            df = df.loc[valid_user_input, ['y', 'pred', 'se', 'iso', 'z_score', 'sector', 'industry']]
            df.insert(0, 'symbol', df.index)
            if not df.empty:
                table = dash_table.DataTable(
                    data=df.sort_values(metric, ascending=False).to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in df.columns] # .drop("color", axis=1)
                )
                return table
        else:
            print('Input not in index')
            pass


# Run App
if __name__ == '__main__':
    app.run_server(debug=True)
