import dash
import json
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import tempfile
import base64
from sklearn.datasets import load_iris
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Define colors for different classes
colors = {0: 'red', 1: 'green', 2: 'blue'}  # You can adjust the colors as needed

app = dash.Dash(__name__)

# Define visualization options for the dropdown
visualization_options = [
    {'label': 'Scatter Plot', 'value': 'scatter-plot'},
    {'label': 'Histogram with PDE', 'value': 'histogram'},
    {'label': 'Box Plot', 'value': 'box-plot'},
    {'label': 'Pair Plot', 'value': 'pair-plot'},
    {'label': 'Correlation Heatmap', 'value': 'correlation-heatmap'},
    {'label': '3D Scatter Plot', 'value': '3d-scatter-plot'},
    {'label': 'Pairwise Scatter Matrix', 'value': 'pairwise-scatter-matrix'},
    {'label': 'Swarm Plot', 'value': 'swarm-plot'},
    {'label': 'Parallel Coordinates Plot', 'value': 'parallel-coordinates-plot'},
    {'label': 'Radial Plot', 'value': 'radial-plot'},
    {'label': 'Cluster Analysis', 'value': 'cluster-analysis'},
]


# Define the app layout
app.layout = html.Div([
    html.H1("Iris Dataset Visualization"),
    html.Label("Select Visualization:"),

    # Dropdown for selecting visualization type
    dcc.Dropdown(
        id='visualization-dropdown',
        options=visualization_options,
        value='scatter-plot',
        multi=False
    ),
    
    # Visualization container (will be populated based on selection)
    html.Div(id='visualization-container')
])


# Create a function for Matplotlib pair plot
def plot_pair_plot():
    sns.set(style="ticks")
    
    # Use the 'hue' parameter to map the target column to colors
    pair_plot = sns.pairplot(df, hue="target", palette=colors)
    
    # Save the Matplotlib figure to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        pair_plot.savefig(tmpfile)
        tmpfile.seek(0)
        # Convert the image to base64 for display
        image_base64 = base64.b64encode(tmpfile.read()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"

# Create a function for the correlation heatmap
def plot_correlation_heatmap():
    corr_matrix = df.corr()
    fig = px.imshow(corr_matrix, x=corr_matrix.columns, y=corr_matrix.columns)
    fig.update_layout(title='Correlation Heatmap')
    return fig

# Create a function for the 3D scatter plot
def plot_3d_scatter_plot():
    fig = px.scatter_3d(df, x='sepal length (cm)', y='sepal width (cm)', z='petal length (cm)', color='target', labels={'target': 'Species'})
    fig.update_layout(title='3D Scatter Plot')
    jsonStr = fig.to_json()  # Convert Plotly figure to JSON serializable format
    return json.loads(jsonStr)  # Return JSON data

# Create a function for the pairwise scatter matrix using Plotly
def plot_pairwise_scatter_matrix():
    fig = px.scatter_matrix(df, dimensions=df.columns[:-1], color='target', labels={'target': 'Species'})
    fig.update_layout(title='Pairwise Scatter Matrix')
    jsonStr = fig.to_json()  # Convert Plotly figure to JSON serializable format
    return json.loads(jsonStr)  # Return JSON data

# Create a Swarm Plot function
def plot_swarm_plot():
    fig = px.scatter(df, x='target', y='petal length (cm)', color='target', labels={'target': 'Species'})
    fig.update_layout(title='Swarm Plot')
    return dcc.Graph(figure=fig)

# Create a Parallel Coordinates Plot function
def plot_parallel_coordinates():
    fig = px.parallel_coordinates(df, color='target', labels={'target': 'Species'})
    fig.update_layout(title='Parallel Coordinates Plot')
    return dcc.Graph(figure=fig)

# Create a Radial Plot function with lines connecting data points to the center
def plot_radial_plot():
    # Create polar coordinates for each data point and convert 'theta' to degrees
    df['theta'] = np.arctan2(df['sepal width (cm)'], df['sepal length (cm)'])
    min_theta = min(df['theta'])
    max_theta = max(df['theta'])
    df['theta'] = 2*np.pi*(df['theta'] - min_theta) / (max_theta - min_theta)
    min_theta = min(df['theta'])
    max_theta = max(df['theta'])
    print("theta: ", min(df['theta']), max(df['theta']))
    df['theta_deg'] = np.degrees(df['theta'])
    print("theta_deg: ", min(df['theta_deg']), max(df['theta_deg']))
    df['r'] = np.sqrt(df['sepal length (cm)']**2 + df['sepal width (cm)']**2)
    fig = px.scatter_polar(df, r='r', theta='theta_deg', color='target', labels={'target': 'Species'})
    fig.update_traces(marker=dict(size=5), line=dict(width=1, color='gray'))
    fig.update_layout(
        title='Radial Plot with Lines Connecting Data Points to Center',
        showlegend=False,
        polar=dict(radialaxis=dict(showticklabels=False, ticks=''),
                   angularaxis=dict(showticklabels=False, ticks='', dtick=45)  # Customize angular axis ticks
                   ),
    )
    return dcc.Graph(figure=fig)


def plot_cluster_analysis():
    pass
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=0).fit(df[iris.feature_names])
    df['cluster'] = kmeans.labels_
    
    fig = px.scatter_3d(
        df, x="sepal length (cm)", y="sepal width (cm)", z="petal length (cm)",
        color="cluster", labels={"cluster": "Cluster"}
    )
    return dcc.Graph(figure=fig)

# Callback to update the visualization based on dropdown selection
@app.callback(
    Output('visualization-container', 'children'), # Output for the visualization container, update_visualization() is called on any input change
    Input('visualization-dropdown', 'value'),
)

# this function is invoked when the dropdown value changes
def update_visualization(selected_visualization): 
    if selected_visualization == 'scatter-plot':
        fig = px.scatter(df, x='sepal length (cm)', y='sepal width (cm)', color='target', labels={'target': 'Species'})
        fig.update_layout(title='Scatter Plot')
        return dcc.Graph(figure=fig)
    elif selected_visualization == 'histogram':
        hist_data = [df[df['target'] == i]['sepal length (cm)'] for i in range(3)]  
        group_labels = ['setosa', 'versicolor', 'virginica']
        fig = ff.create_distplot(hist_data, group_labels, bin_size=0.2, show_hist=True, show_rug=False)
        fig.update_layout(title='Histogram with PDE (Sepal Length)')
        return dcc.Graph(figure=fig)
    elif selected_visualization == 'box-plot':
        fig = px.box(df, x='target', y='petal width (cm)', labels={'target': 'Species'})
        fig.update_layout(title='Box Plot')
        return dcc.Graph(figure=fig)
    elif selected_visualization == 'pair-plot':
        # Call the Matplotlib function and get the image data
        image_data = plot_pair_plot()
        return html.Img(src=image_data, style={'width': '100%'})
    elif selected_visualization == 'correlation-heatmap':
        fig = plot_correlation_heatmap()
        return dcc.Graph(figure=fig)
    elif selected_visualization == '3d-scatter-plot':
        fig = plot_3d_scatter_plot()
        return dcc.Graph(figure=fig)
    elif selected_visualization == 'pairwise-scatter-matrix':
        pair_plot = plot_pairwise_scatter_matrix()
        return dcc.Graph(figure=pair_plot)
    elif selected_visualization == 'swarm-plot':
        return plot_swarm_plot()
    elif selected_visualization == 'parallel-coordinates-plot':
        return plot_parallel_coordinates()
    elif selected_visualization == 'radial-plot':
        return plot_radial_plot()
    elif selected_visualization == 'bounded-radial-plot':
        return plot_bounded_radial_plot()
    elif selected_visualization == 'cluster-analysis':
        return plot_cluster_analysis()

if __name__ == '__main__':
    app.run_server(debug=True)
