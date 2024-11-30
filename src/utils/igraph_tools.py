import igraph as ig
from typing import Dict, List

class Tools:
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
        def has_kuratowski_subgraph(graph):
            """
            Check if the graph contains a Kuratowski subgraph (K5 or K3,3).

            Parameters:
                graph (igraph.Graph): The graph to check.

            Returns:
                bool: True if a Kuratowski subgraph is found, False otherwise.
            """
            # Check for K5 (complete graph on 5 nodes)
            if len(graph.vs) >= 5 and graph.isomorphic(ig.Graph.Full(5)):
                return True

            # Check for K3,3 (complete bipartite graph with 3 nodes in each set)
            if len(graph.vs) >= 6 and graph.isomorphic(ig.Graph.Full_Bipartite(3, 3)):
                return True

            return False

        def is_planar_recursive(graph):
            """
            Recursive function to check if a graph is planar.

            Parameters:
                graph (igraph.Graph): The graph to check.

            Returns:
                bool: True if the graph is planar, False otherwise.
            """
            if len(graph.vs) <= 4:  # All graphs with 4 or fewer vertices are planar
                return True

            if has_kuratowski_subgraph(graph):
                return False

            for edge in graph.es:
                graph_copy = graph.copy()
                graph_copy.contract_vertices([edge.source, edge.target])  # Contract an edge

                if not is_planar_recursive(graph_copy):
                    return False

            return True

        return is_planar_recursive(igraph_graph)
    
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
    

    def calculate_sparsity(graph: ig.Graph) -> float:
        """
        Calculate the sparsity of a graph.
        
        Parameters:
            graph (igraph.Graph): The input graph.
        
        Returns:
            float: The sparsity of the graph.
        """
        num_vertices = graph.vcount()
        num_edges = graph.ecount()
        
        if num_vertices <= 1:
            return 1.0  # A graph with 0 or 1 vertices is trivially sparse
        
        max_edges = num_vertices * (num_vertices - 1)
        edge_density = num_edges / max_edges
        sparsity = 1 - edge_density
        
        return sparsity
