import numpy as np
import igraph as ig

from typing import Dict, List

class GraphGenerator:
    """
    A class to generate various types of graphs using igraph.
    """

    @staticmethod
    def generate_erdos_renyi(n, m=None, p=None) -> ig.Graph:
        """
        Generate an Erdős-Rényi random graph.

        Parameters:
            n (int): Number of vertices.
            m (int, optional): Number of edges. If specified, it overrides p.
            p (float, optional): Probability of edge creation. Ignored if m is provided.

        Returns:
            igraph.Graph: The generated graph.
        """
        if m is not None:
            return ig.Graph.Erdos_Renyi(n=n, m=m, directed=False, loops=False)
        elif p is not None:
            return ig.Graph.Erdos_Renyi(n=n, p=p, directed=False, loops=False)
        else:
            raise ValueError("Either m or p must be provided.")

    @staticmethod
    def generate_small_world(n, k, p) -> ig.Graph:
        """
        Generate a Small-World graph using the Watts-Strogatz model.

        Parameters:
            n (int): Number of vertices.
            k (int): Each vertex is connected to k nearest neighbors in a ring topology.
            p (float): Probability of rewiring each edge.

        Returns:
            igraph.Graph: The generated graph.
        """
        return ig.Graph.Watts_Strogatz(dim=1, size=n, nei=k//2, p=p)

    @staticmethod
    def generate_barabasi_albert(n, m) -> ig.Graph:
        """
        Generate a Barabási-Albert graph with preferential attachment.

        Parameters:
            n (int): Number of vertices.
            m (int): Number of edges to attach from a new node to existing nodes.

        Returns:
            igraph.Graph: The generated graph.
        """
        return ig.Graph.Barabasi(n=n, m=m)

    @staticmethod
    def generate_stochastic_block_model(n, sizes, p_matrix) -> ig.Graph:
        """
        Generate a Stochastic Block Model (SBM) graph.

        Parameters:
            n (int): Total number of vertices.
            sizes (list of int): Sizes of each block (community).
            p_matrix (list of list of float): Matrix of edge probabilities between blocks.

        Returns:
            igraph.Graph: The generated graph.
        """
        return ig.Graph.SBM(n = n, block_sizes=sizes, pref_matrix=p_matrix)

