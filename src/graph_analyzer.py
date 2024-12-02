import os
while not os.getcwd().endswith('project_2024'):
    print(os.getcwd())
    os.chdir('../')

import numpy as np
import igraph as ig
import distfit as df
from PIL import Image
import planarity as pl
from scipy import stats
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
        results["avg_degree"] = np.mean(graph.degree())
        results["planarity"] = cls.is_graph_planar(graph)
        results["connectedness"] = graph.is_connected()
        results["self_loops"] = cls.has_self_loops(graph)
        results["density"] = graph.density(results["self_loops"])
        results["degree_centrality"] = cls.calculate_centrality_measures(graph, 'degree')
        results["betweenness_centrality"] = cls.calculate_centrality_measures(graph, 'betweenness')
        results["closeness_centrality"] = cls.calculate_centrality_measures(graph, 'closeness')
        results["eigenvector_centrality"] = cls.calculate_centrality_measures(graph, 'eigenvector')


        if graph.is_directed():
            results["local_clustering_coeffs"]  = graph.transitivity_local_directed()
            results["clustering_coeff"]  = graph.transitivity_avglocal_directed()      
        else:
            results["local_clustering_coeffs"] = graph.transitivity_local_undirected()
            results["clustering_coeff"] = graph.transitivity_undirected()
        
        results["nr_of_trinagles"] = len(graph.cliques(min=3, max=3))
        results["communities_infomap"] = cls.detect_communities(graph, 'infomap')

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
        # assert (len(vertex_color) == 1) or (len(vertex_color) == graph.vcount()),  "vertex_color must match the number of vertices."
        # assert (len(edge_color) == 1) or (len(edge_color) == graph.ecount()), "edge_color must match the number of edges."
        # assert (len(edge_width) == 1) or (len(edge_width) == graph.ecount()), "edge_width must match the number of edges."
        # assert (vertex_label is None) or (isinstance(vertex_label, List) and len(vertex_label) == graph.vcount()), "vertex_label must match the number of vertices."
        
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
            axes[i].set_title(os.path.splitext(image_file)[0], fontsize=8)

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
    
    @classmethod
    def calculate_centrality_measures(cls, graph: ig.Graph, method: str):
        """
        Calculate and print all centrality measures implemented in igraph.

        Parameters:
            graph (igraph.Graph): The input graph for centrality calculation.
            method (str): The centrality measure to calculate. Supported methods are:
                - 'centrality': Degree centrality.
                - 'closeness': Closeness centrality.
                - 'betweenness': Betweenness centrality.
                - 'eigenvector': Eigenvector centrality.
                - 'pagerank': PageRank centrality.
                - 'harmonic': Harmonic centrality.
                - 'load': Load centrality.
                - 'katz': Katz centrality.

        Returns:
            dict: A dictionary containing centrality measures for all nodes.
        """
        try:
            if method == 'degree':
                centrality = graph.degree()
            elif method == 'closeness':
                centrality = graph.closeness(normalized=True)
            elif method == 'betweenness':
                centrality = graph.betweenness(vertices=None, directed=False, weights=None, cutoff=None)
            elif method == 'eigenvector':
                centrality = graph.eigenvector_centrality(directed=False, scale=True, weights=None, return_eigenvalue=False)
            elif method == 'pagerank':
                centrality = graph.pagerank(directed=False, damping=0.85, weights=None, arpack_options=None)
            elif method == 'harmonic':
                centrality = graph.harmonic_centrality(vertices=None, mode='ALL', weights=None)
            elif method == 'load':
                centrality = cls.calculate_load_centrality(graph)
            elif method == 'katz':
                centrality = cls.calculate_katz_centrality(graph)
            else:
                raise ValueError(f"Unsupported centrality method: '{method}'. Available methods: degree, closeness, betweenness, eigenvector, pagerank, harmonic, load, katz.")
            
            return centrality
        
        except Exception as e:
            raise ValueError(f"Error computing centrality for method '{method}': {str(e)}")

    
    @staticmethod
    def plot_adjacency_matrix(graph, 
                              node_quantity: np.ndarray = None, 
                              label: str = 'Non', 
                              lang: str = 'eng', 
                              format: str = 'png', 
                              block: int = 0, 
                              mem: np.ndarray = np.array([None]), 
                              no_ticks: bool = 0):
        """
        Plots the adjacency matrix of a given graph, with optional clustering and localization color overlay.

        Parameters:
            graph (igraph.Graph): Input graph object.
            node_quantity (numpy.ndarray): Node quantity for which cmap to be done.
            label (str): Label for the plot title and saved file (default: 'Non').
            lang (str): Language for the plot title ('eng' for English, 'hu' for Hungarian) (default: 'eng').
            format (str): Format to save the plot (e.g., 'png', 'jpg') (default: 'png').
            block (int): If non-zero, enables block plotting with clustering (default: 0).
            mem (numpy.ndarray): Membership array for clustering (default: array([None])).
            no_ticks (bool): If non-zero, removes axis ticks (default: 0).

        Returns:
            tuple: Contains the x and y coordinates, sorted indices, membership, and connected nodes.
        """
        colormap = plt.cm.copper

        def save_plot(title, file_name):
            plt.title(title)
            plt.tight_layout()
            plt.savefig(file_name, format=format, dpi = 300)
            plt.show()

        def setup_plot(matrix_size):
            plt.figure(figsize=(6, 6))
            plt.xlim(0, matrix_size)
            plt.ylim(0, matrix_size)
            if no_ticks:
                plt.xticks([])
                plt.yticks([])
            plt.gca().set_aspect('equal')
            

        if block and mem[0] is not None:
            print('Block mode enabled')

            # Perform clustering and sort adjacency matrix
            comm = ig.VertexClustering(graph, membership=mem)
            graph.clustering = comm
            membership = np.array(comm.membership)
            sorted_indices = np.argsort(membership).astype(int)
            adjacency_matrix = np.array(graph.get_adjacency().data)
            adjacency_matrix = adjacency_matrix[sorted_indices][:, sorted_indices]

            # Prepare coordinates and connections
            matrix_size = adjacency_matrix.shape[0]
            x_coords, y_coords = np.meshgrid(range(matrix_size), range(matrix_size))
            x = x_coords.flatten()
            y = y_coords.flatten()
            connections = adjacency_matrix.flatten()
            connected_nodes = np.where(connections > 0)[0]
            setup_plot(matrix_size)

            # Color normalization based on localization color overlay (LCO)
            if node_quantity.any():
                col_vec = 1 - abs(node_quantity[x[connected_nodes]] - node_quantity[y[connected_nodes]])
                norm = plt.Normalize(vmin=np.min(col_vec), vmax=np.max(col_vec))
                scatter = plt.scatter(x[connected_nodes], y[connected_nodes], c=col_vec, cmap=colormap, norm=norm, s=2.0)
            else:
                scatter = plt.scatter(x[connected_nodes], y[connected_nodes], color='red', s=2.0)

            # Add colorbar
            cbar = plt.colorbar(scatter, label='LCO')
            cbar.ax.tick_params(labelsize=15)
            cbar.set_label('LCO', fontsize=25)


            # Save plot with language-specific title
            if lang == 'eng':
                save_plot(f'Adjacency Matrix of {label}', f'adjacency_matrix_eng_{label}_clust.{format}')
            elif lang == 'hu':
                save_plot(f'{label} hálózat szomszédsági mátrixa', f'adjacency_matrix_hu_{label}_clust.{format}')
            else:
                save_plot(f'Adjacency Matrix {label}', f'adjacency_matrix_{label}_clust.{format}')

            return x, y, sorted_indices, membership, connected_nodes

        else:
            # Non-clustering mode
            # Extract adjacency matrix
            adjacency_matrix = np.array(graph.get_adjacency().data)
            matrix_size = adjacency_matrix.shape[0]
            x_coords, y_coords = np.meshgrid(range(matrix_size), range(matrix_size))
            x = x_coords.flatten()
            y = y_coords.flatten()
            connections = adjacency_matrix.flatten()
            connected_nodes = np.where(connections > 0)[0]
            setup_plot(matrix_size)

            # Scatter plot of adjacency matrix
            plt.scatter(x[connected_nodes], y[connected_nodes], color='red', s=1.0)


            # Save plot with language-specific title
            if lang == 'eng':
                save_plot(f'Adjacency Matrix of {label}', f'adjacency_matrix_eng_{label}.{format}')
            elif lang == 'hu':
                save_plot(f'{label} hálózat szomszédsági mátrixa', f'adjacency_matrix_hu_{label}.{format}')
            else:
                save_plot(f'Adjacency Matrix {label}', f'adjacency_matrix_{label}.{format}')

            return x, y, None, None, connected_nodes

    @staticmethod
    def detect_communities(graph, method=None, **kwargs):
        """
        Apply various community detection algorithms on a graph and return results.

        Parameters:
            graph (igraph.Graph): The input graph for community detection.
            method (str): The specific community detection method to use (optional).
            **kwargs: Additional parameters to customize community detection algorithms.
                - directed (bool): Whether the graph is directed.
                - weights (list or None): Edge weights.
                - vertex_weights (list or None): Vertex weights.
                - steps (int): Number of steps for the Walktrap algorithm.
                - trials (int): Number of trials for the Infomap algorithm.
                - resolution (float): Resolution parameter for the Multilevel algorithm.
                - spins (int): Number of spins for the Spinglass algorithm.

        Returns:
            dict or list: A dictionary with method names as keys and membership lists as values,
                          or a membership list if a specific method is provided.

        Raises:
            ValueError: If the specified method is not supported or fails.
        """
        results = {}

        def run_method(name, func):
            try:
                return func()
            except Exception as e:
                return f"Failed: {e}"

        directed = kwargs.get('directed', False)
        weights = kwargs.get('weights', None)
        vertex_weights = kwargs.get('vertex_weights', None)
        steps = kwargs.get('steps', 4)
        trials = kwargs.get('trials', 10)
        resolution = kwargs.get('resolution', 1.0)
        spins = kwargs.get('spins', 25)

        methods = {
            'edge_betweenness': lambda: graph.community_edge_betweenness(directed=directed, weights=weights).as_clustering().membership,
            'walktrap': lambda: graph.community_walktrap(weights=weights, steps=steps).as_clustering().membership,
            'infomap': lambda: graph.community_infomap(edge_weights=weights, vertex_weights=vertex_weights, trials=trials).membership,
            'label_propagation': lambda: graph.community_label_propagation(weights=weights, initial=None, fixed=None).membership,
            'leading_eigenvector': lambda: graph.community_leading_eigenvector(weights=weights, clusters=None, arpack_options=None).membership,
            'multilevel': lambda: graph.community_multilevel(weights=weights, resolution=resolution, return_levels=False).membership,
            'fast_greedy': lambda: graph.community_fastgreedy(weights=weights).as_clustering().membership,
            'spinglass': lambda: graph.community_spinglass(weights=weights, spins=spins, parupdate=False, start_temp=1.0, stop_temp=0.01, gamma=1.0, update_rule="simple").membership,
            'optimal_modularity': lambda: graph.community_optimal_modularity(weights=weights).membership
        }

        if method:
            if method not in methods:
                raise ValueError(f"Unsupported community detection method: '{method}'. Available methods: {', '.join(methods.keys())}.")
            return run_method(method, methods[method])

        for name, func in methods.items():
            results[name] = run_method(name, func)

        return results
    
    @staticmethod
    def histogram(data, 
               bins: int = 100, 
               savename: str = None, 
               show: bool = 1, 
               logx: bool = 0, 
               logy: bool = 0,
               labelx: str = r'$x$',
               labely: str = r'$P(x)$', 
               density: bool = True,
               distfit_enabled: bool = False,
               distfit_distr: List[str] = ['norm', 'expon', 'pareto', 't', 'genextreme', 'gamma', 'lognorm', 'beta', 'loggamma'],
               data_label: str = 'Data',
               bar_hist: bool = False):
        
        """
        Plot a histogram for the given data.

        Parameters:
            data (array-like): Input data to be plotted.
            bins (int): Number of bins in the histogram (default: 100).
            savename (str): File name to save the plot. If None, the plot is not saved (default: None).
            show (bool): Whether to display the plot (default: True).
            logx (bool): Apply logarithmic scaling to the x-axis (default: False).
            logy (bool): Apply logarithmic scaling to the y-axis (default: False).
            labelx (str): Label for the x-axis (default: r'$\omega$').
            labely (str): Label for the y-axis (default: r'$p(\omega)$').
            density (bool): Normalize the histogram to have a total area of 1 (default: True).
            distfit_enabled (bool): Whether to enable distfit functionality (default: False).
            distfit_distr (list): List of distributions for distfit to try (default: None, uses default distfit distributions).
                All distributions:
                    ['alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'burr', 'cauchy', 'chi', 
                    'chi2', 'cosine', 'dgamma', 'erlang', 'expon', 'exponnorm', 'exponweib', 
                    'exponpow', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'genlogistic', 
                    'genpareto', 'gennorm', 'genexpon', 'genextreme', 'gausshyper', 'gamma', 'gengamma', 
                    'genhalflogistic', 'gibrat', 'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy', 
                    'halflogistic', 'halfnorm', 'halfgennorm', 'hypsecant', 'invgamma', 'invgauss',
                    'johnsonsb', 'johnsonsu', 'laplace', 'levy', 'logistic', 'loggamma', 
                    'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'nakagami', 'norm', 'pareto', 
                    'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'reciprocal', 'rayleigh', 
                    'rice', 'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon', 'truncnorm', 
                    'uniform', 'vonmises', 'vonmises_line', 'wald', 'wrapcauchy']
            data_label (str): Label of ythe input data
            bar_hist (bool): Use bar-style histogram instead of line-style (default: False).
        """
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        # exclude NaN values
        data = data[~np.isnan(data)]

        plt.figure(figsize=(10, 8))
        
        # Compute histogram
        hist, bin_edges = np.histogram(data, bins=bins, density=density)
        
        # Plot the histogram
        if bar_hist:
            plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), 
                    align='edge', color='r', edgecolor='k', label = data_label)
        else:
            plt.plot(bin_edges[:-1], hist, 'ko-', mec='k', mfc ='r', mew=2, lw=2, label = data_label)
        
        # Distfit functionality
        if distfit_enabled:
            dist = df.distfit(distr=distfit_distr)
            dist.fit_transform(data, verbose=0)
            dist.predict(data, verbose=0)
            best_fit_name = dist.model['name']
            best_fit_params = dist.model['params']

            # Generate fitted curve
            x_fit = np.linspace(min(data), max(data), 1000)
            arg = best_fit_params[:-2]
            loc = best_fit_params[-2]
            scale = best_fit_params[-1]
            distribution = getattr(stats, best_fit_name)
            y_fit = distribution.pdf(x_fit, loc=loc, scale=scale, *arg)

            # Plot distfit curve
            plt.plot(x_fit, y_fit, 'limegreen', lw=2, label=f'Fit: {best_fit_name} (loc={round(loc, 2)}, scale={round(scale, 2)})')

        # Apply logarithmic scales if specified
        if logx:
            plt.gca().set_xscale("log")
        if logy:
            plt.yscale('log')
        
        # Set axis labels
        plt.xlabel(labelx)
        plt.ylabel(labely)
        plt.legend()
        plt.tight_layout()
        # Save the plot if a file name is provided
        if savename:
            plt.savefig(savename, dpi=300)
        
        # Show or close the plot
        if show:
            plt.show()
        else:
            plt.ioff()
            plt.close()
        

