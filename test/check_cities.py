import os
import pandas as pd

def load_city_graph_data(folder_path):
    """
    Load and print the number of edges and nodes for each city from CSV files.

    Parameters:
        folder_path (str): Path to the folder containing CSV files.
    
    Returns:
        None
    """
    # List all files in the folder
    files = os.listdir(folder_path)

    # Group files by city name
    edges_files = [f for f in files if f.endswith('_edges.csv')]
    nodes_files = [f for f in files if f.endswith('_nodes.csv')]

    # Create a dictionary to match city names with edges and nodes files
    city_data = {}
    for file in edges_files:
        city_name = file.replace('_edges.csv', '')
        city_data[city_name] = {'edges': os.path.join(folder_path, file)}

    for file in nodes_files:
        city_name = file.replace('_nodes.csv', '')
        if city_name in city_data:
            city_data[city_name]['nodes'] = os.path.join(folder_path, file)
        else:
            city_data[city_name] = {'nodes': os.path.join(folder_path, file)}

    # Load each city's data and print counts
    for city, files in city_data.items():
        edges_count = 0
        nodes_count = 0

        if 'edges' in files:
            edges_df = pd.read_csv(files['edges'])
            edges_count = len(edges_df)

        if 'nodes' in files:
            nodes_df = pd.read_csv(files['nodes'])
            nodes_count = len(nodes_df)

        print(f"City: {city}")
        print(f"  Nodes: {nodes_count}")
        print(f"  Edges: {edges_count}\n")

if __name__ == "__main__":
    folder_path = "road_nets"  # Replace with the path to your folder
    load_city_graph_data(folder_path)
