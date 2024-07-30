import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
import networkx as nx
import plotly.graph_objs as go
import pickle
import os
import sys
sys.path.append(os.path.abspath('model/'))
from utils import create_hetero_data

# Variables need specification
data_name = 'amlsim_mixed'  # 'elliptic' dgraph_fin amlsim_mixed
ext_rate = 0.3
model_prediction = np.load(f'outputs/{data_name}/hetero_SAGE-SMOTE-ReNode_ext_{ext_rate}.npz')
# Define the layout, e.g. PCA or TSNE, see the notebook
with open(f'dashboard/{data_name}_ext_{ext_rate}_layout.pkl', 'rb') as f:
    pos = pickle.load(f)



# Create the hetero data
data_path = f'hetero_data/{data_name}/ext_{ext_rate}/'
data = create_hetero_data(data_path)
accounts = pd.read_csv(data_path + 'accounts.csv')
idx_internal = accounts['internal'] == True



# Prepare data for visualization
internal_ids_mapped = data['internal'].id.numpy()
external_ids_mapped = data['external'].id.numpy()
internal_ids_real = accounts['account_id'][idx_internal].values.astype(str)
external_ids_real = accounts['account_id'][~idx_internal].values.astype(str)

# Map the node ID back to the original node ID by creating a dictionary
internal_id_map = dict(zip(internal_ids_mapped, internal_ids_real))
external_id_map = dict(zip(external_ids_mapped, external_ids_real))

# Extract edge indices
internal_internal_edges = data['internal', 'internal_txn', 'internal'].edge_index.numpy()
internal_external_edges = data['internal', 'external_withdraw', 'external'].edge_index.numpy()
external_internal_edges = data['external', 'external_deposit', 'internal'].edge_index.numpy()

# Map the edge indices back to the original node ID, edge[0] and edge[1] are the source and target nodes, different nodes type
internal_internal_edges = np.vectorize(internal_id_map.get)(internal_internal_edges)
internal_external_edges = np.array([
    [internal_id_map[src], external_id_map[dst]] 
    for src, dst in internal_external_edges.T
]).T

external_internal_edges = np.array([
    [external_id_map[src], internal_id_map[dst]] 
    for src, dst in external_internal_edges.T
]).T

# Create a bipartite graph using NetworkX
B = nx.Graph()

# Add nodes with the bipartite attribute
B.add_nodes_from(internal_ids_real, bipartite=0, type='internal')
B.add_nodes_from(external_ids_real, bipartite=1, type='external')

# Add edges (assuming you want to visualize both internal to external and external to internal)
for edge in internal_external_edges.T:
    B.add_edge(edge[0], edge[1])

for edge in external_internal_edges.T:
    B.add_edge(edge[0], edge[1])

for edge in internal_internal_edges.T:
    B.add_edge(edge[0], edge[1])
print('Graph has been created')
# Create a mapping from real node ID to its position
pos_dict = {**{internal_ids_real[i]: pos[i] for i in range(len(internal_ids_real))},
            **{external_ids_real[i]: pos[len(internal_ids_real) + i] for i in range(len(external_ids_real))}}

# Prepare the fraud probability for each internal node
pred_label = np.argmax(model_prediction['pred_scores'], axis=1)  # Predicted label for fraud or not
assert len(internal_ids_real) == len(pred_label)

# Function to get top N neighbors based on degree (number of connections)
def get_top_n_neighbors(graph, node, top_n):
    neighbors = list(graph.neighbors(node))
    sorted_neighbors = sorted(neighbors, key=lambda x: len(list(graph.neighbors(x))), reverse=True)
    return sorted_neighbors[:top_n]

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Graph Visualization of Internal and External Accounts Transactions"),
    html.Label('Select Internal Account:'),
    dcc.Dropdown(
        id='internal-account-dropdown',
        options=[{'label': str(acc), 'value': acc} for acc in internal_ids_real],
        value=internal_ids_real[0]
    ),
    html.Label('Select Top N Neighbors:'),
    dcc.Slider(
        id='top-n-slider',
        min=1,
        max=10,
        step=1,
        value=5,
        marks={i: str(i) for i in range(1, 11)}
    ),
    html.Label('Data to Display:'),
    dcc.RadioItems(
        id='data-display-radio',
        options=[
            {'label': 'Show Real Data', 'value': 'real'},
            {'label': 'Show Prediction', 'value': 'predicted'}
        ],
        value='real'
    ),
    html.Label('Show Account Names:'),
    dcc.Checklist(
        id='show-text-checkbox',
        options=[{'label': 'Show Text', 'value': 'show'}],
        value=[]
    ),
    
    html.Div([
    html.Div(style={"width": "10px", "height": "10px", "background-color": "blue", "border-radius": "50%", "display": "inline-block", "margin-right": "5px"}),
    html.Span("Internal", style={"margin-right": "10px"}),
    html.Div(style={"width": "10px", "height": "10px", "background-color": "red", "border-radius": "50%", "display": "inline-block", "margin-right": "5px"}),
    html.Span("Fraud", style={"margin-right": "10px"}),
    html.Div(style={"width": "10px", "height": "10px", "background-color": "green", "border-radius": "50%", "display": "inline-block", "margin-right": "5px"}),
    html.Span("External", style={"margin-right": "10px"})
], style={"display": "flex", "justify-content": "center", "margin-bottom": "20px"}),
    
    dcc.Graph(id='graph-plot')
])

@app.callback(
    Output('graph-plot', 'figure'),
    [Input('internal-account-dropdown', 'value'),
     Input('top-n-slider', 'value'),
     Input('data-display-radio', 'value'),
     Input('show-text-checkbox', 'value')]
)
def update_graph(selected_account, top_n, data_display, show_text):
    top_n_neighbors = get_top_n_neighbors(B, selected_account, top_n)

    edge_x = []
    edge_y = []
    nodes_to_include = {selected_account}
    for neighbor in top_n_neighbors:
        nodes_to_include.add(neighbor)
        x0, y0 = pos_dict[selected_account]
        x1, y1 = pos_dict[neighbor]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Also add edges from neighbor to its top N neighbors if you want more hop interactions
        neighbors_of_neighbor = get_top_n_neighbors(B, neighbor, top_n)
        for second_neighbor in neighbors_of_neighbor:
            nodes_to_include.add(second_neighbor)
            x0, y0 = pos_dict[neighbor]
            x1, y1 = pos_dict[second_neighbor]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='blue'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_color = []
    node_text = []
    for node in nodes_to_include:
        x, y = pos_dict[node]
        node_x.append(x)
        node_y.append(y)
        if B.nodes[node]['type'] == 'internal':
            if data_display == 'real':
                fraud_label = data['internal'].y[np.where(internal_ids_real == node)[0][0]]
            else:
                fraud_label = pred_label[np.where(internal_ids_real == node)[0][0]]
            
            if fraud_label == 1:
                node_color.append('red')  # Color for fraud nodes
            else:
                node_color.append('blue')  # Color for non-fraud nodes
            
            node_text.append(f'Internal Node {node}, Fraud Alert: {fraud_label}')
        else:
            node_color.append('green')
            node_text.append(f'External Node {node}')

        if 'show' in show_text:
            node_text[-1] = node  # Show the ID as the text

    mode = 'markers+text' if 'show' in show_text else 'markers'
    hoverinfo = 'none' if 'show' in show_text else 'text'

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode=mode,
        textposition="top center",
        hoverinfo=hoverinfo,
        marker=dict(
            showscale=False,
            colorscale=[[0, 'green'], [0.5, 'blue'], [1, 'red']],
            color=node_color,
            size=10,
            # colorbar=dict(
            #     thickness=15,
            #     title='Node Type',
            #     xanchor='left',
            #     titleside='right',
            #     tickvals=[0, 1, 2],
            #     ticktext=['External', 'Internal', 'Fraud']
            # ),
            line_width=2),
        text=node_text
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Graph Visualization',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

if __name__ == '__main__':   
    app.run_server(debug=True)