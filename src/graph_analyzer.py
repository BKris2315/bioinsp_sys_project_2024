import os
while not os.getcwd().endswith('project_2024'):
    print(os.getcwd())
    os.chdir('../')

import time
import numpy as np
import igraph as ig
from PIL import Image
import networkx as nx
from tqdm import tqdm
import planarity as pl
import matplotlib.pyplot as plt
from typing import Dict, Any, List
class NetworkAnalyzer:
    """
    A class to analyze network properties using igraph.
    """
    
    @classmethod
    def get_basic_properties(cls, graph: ig.Graph) -> Dict[str, Any]:
        results = {}
        results["edge_nr"] = graph.ecount()
        results["vertices_nr"] = graph.vcount()
        results["degree"] = graph.degree()
        results["avg_degree"] = np.mean(graph.degree())
        results["planarity"] = cls.is_graph_planar(graph)
        results["connectedness"] = graph.is_connected()
        results["self_loops"] = cls.has_self_loops(graph)
        results["density"] = graph.density(results["self_loops"])

        if graph.is_directed():
            results["local_clustering_coeffs"]  = graph.transitivity_local_directed()
            results["clustering_coeff"]  = graph.transitivity_avglocal_directed()      
        else:
            results["local_clustering_coeffs"] = graph.transitivity_local_undirected()
            results["clustering_coeff"] = graph.transitivity_undirected()
        
        results["nr_of_trinagles"] = len(graph.cliques(min=3, max=3))

        return results
    
    @staticmethod
    def is_graph_planar(igraph_graph: ig.Graph) -> bool:
        """
        Check if a graph is planar without using NetworkX.
        This implementation uses the Kuratowski's theorem-based approach.

        Parameters:
            igraph_graph (igraph.Graph): The igraph graph to be checked.

        Returns:
            bool: True if the graph is planar, False otherwise.
        """
        return pl.is_planar(igraph_graph.get_edgelist())
    
    @staticmethod
    def has_self_loops(igraph_graph: ig.Graph) -> bool:
        """
        Check if a graph has self-loops.

        Parameters:
            igraph_graph (igraph.Graph): The graph to check.

        Returns:
            bool: True if the graph has self-loops, False otherwise.
        """
        for edge in igraph_graph.es:
            if edge.source == edge.target:
                return True
        return False
    
    @staticmethod    
    def plot_graph(graph: ig.Graph, layout: str = 'kamada_kawai', 
                   vertex_size: float = [10],
                   vertex_color: List[str] = ['red'],
                   edge_color: List[str] = ['black'],
                   edge_width: List[float] = [1],
                   vertex_label: List[str] = None,
                   bbox: List[int] = [600,600], 
                   edge_curved: bool = True,
                   savename: str = None) -> ig.drawing.cairo.plot.CairoPlot:
        """
        Plot the graph using the specified layout and save it as an image if a filename is provided.

        Parameters:
            graph (igraph.Graph): The graph to be plotted.
            layout (str): The layout algorithm to use (e.g., 'kamada_kawai', 'fruchterman_reingold', etc.).
            vertex_size (float or List[float]): Size of vertices. Either a single value or a list matching the number of vertices.
            vertex_color (List[str]): Colors for vertices. Either a single color or a list matching the number of vertices.
            edge_color (List[str]): Colors for edges. Either a single color or a list matching the number of edges.
            edge_width (float or List[float]): Width of edges. Either a single value or a list matching the number of edges.
            vertex_label (List[str], optional): Labels for vertices. Either None or a list matching the number of vertices.
            bbox (List[int], optional): Bounding box dimensions for the plot. Defaults to [800,800].
            edge_curved (bool, optional): Plot edges in curved fashion. Defaults to True
            savename (str, optional): File path to save the plot image. If None, the plot is not saved.

        Returns:
            ig.CairoPlot: The generated plot object.
        """
        assert (len(vertex_color) == graph.vcount()) or (len(vertex_color) == 1),  "vertex_color must match the number of vertices."
        assert (len(edge_color) == graph.ecount()) or (len(edge_color) == 1), "edge_color must match the number of edges."
        assert (len(edge_width) == graph.ecount()) or (len(edge_width) == 1), "edge_width must match the number of edges."
        assert (vertex_label is None) or (isinstance(vertex_label, List) and len(vertex_label) == graph.vcount()), "vertex_label must match the number of vertices."
        
        if layout is None:
            pos = graph.layout_auto()
        else:
            pos = graph.layout(layout)
            
        fig = ig.plot(graph, layout=pos, 
                vertex_size=vertex_size, 
                vertex_color=vertex_color,
                edge_color=edge_color,
                edge_width=edge_width,
                vertex_label=vertex_label,
                bbox = bbox,
                edge_curved=edge_curved)
        if savename:
            fig.save(savename)
        return fig

    @staticmethod
    def plot_images_from_folder(folder_path: str, 
                                subplot_ratio: tuple = (4, 3),
                                savename: str = None):
        """
        Load images from a folder and plot them as subplots in a specified ratio.

        Parameters:
            folder_path (str): Path to the folder containing image files.
            subplot_ratio (tuple): Ratio of rows to columns (e.g., (4, 3)).

        Returns:
            None
        """
        # Get all image files in the folder
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print("No images found in the folder.")
            return

        # Calculate rows and columns for subplots
        num_images = len(image_files)
        rows, cols = subplot_ratio
        
        # Adjust subplot dimensions if needed
        total_subplots = rows * cols
        while total_subplots < num_images:
            rows += 1
            total_subplots = rows * cols

        # Create the figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), dpi=150)
        axes = axes.flatten()  # Flatten the axes for easier iteration

        # Load and display images
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path)
            
            # Display image
            axes[i].imshow(image)
            axes[i].axis('off')
            axes[i].set_title(image_file, fontsize=8)

        # Hide remaining empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        if savename:
            plt.savefig(savename, dpi = 300)
        plt.show()

    @staticmethod
    def calculate_katz_centrality(graph: ig.Graph, 
                                  alpha:float = 0.1, 
                                  beta: float = 1.0, 
                                  tol: float = 1e-9, 
                                  maxiter: int = 1000):
        """
        Calculate Katz centrality for all nodes in the graph.

        Parameters:
            graph (igraph.Graph): The input graph.
            alpha (float): Attenuation factor.
            beta (float): Weight for the centrality contribution.
            tol (float): Convergence tolerance.
            maxiter (int): Maximum number of iterations.

        Returns:
            list: Katz centrality values for all nodes.
        """
        adjacency_matrix = graph.get_adjacency().data
        num_nodes = len(adjacency_matrix)

        # Initialize centrality scores
        katz_centrality = [1.0] * num_nodes

        for _ in range(maxiter):
            new_centrality = [beta + alpha * sum(adjacency_matrix[j][i] * katz_centrality[j] for j in range(num_nodes))
                            for i in range(num_nodes)]
            diff = sum(abs(new_centrality[i] - katz_centrality[i]) for i in range(num_nodes))
            katz_centrality = new_centrality
            if diff < tol:
                break

        return katz_centrality
    
    @staticmethod
    def calculate_load_centrality(graph: ig.Graph):
        """
        Calculate load centrality for all nodes in the graph.

        Parameters:
            graph (igraph.Graph): The input graph.

        Returns:
            list: Load centrality values for all nodes.
        """

        # Initialize load centrality to zero
        load_centrality = [0.0] * graph.vcount()

        # Iterate over all pairs of vertices to calculate load
        for source in graph.vs:
            for target in graph.vs:
                if source != target:
                    paths = graph.get_all_shortest_paths(source, to=target)
                    total_paths = len(paths)
                    if total_paths > 0:
                        for path in paths:
                            for node in path[1:-1]:  # Exclude source and target
                                load_centrality[node] += 1 / total_paths

        return load_centrality

