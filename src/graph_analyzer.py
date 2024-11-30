import os
while not os.getcwd().endswith('project_2024'):
    print(os.getcwd())
    os.chdir('../')

import time
import numpy as np
import igraph as ig
import networkx as nx

from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from src.utils.igraph_tools import Tools as tl

class NetworkAnalyzer:
    def __init__(self, graph: ig.Graph):
        self.graph = graph

    def get_basic_properties(self) -> Dict[str, Any]:
        results = {}
        results["edge_nr"] = self.graph.ecount()
        results["vertices_nr"] = self.graph.vcount()
        results["degree"] = self.graph.degree()
        results["avg_degree"] = np.mean(self.graph.avg_degree())
        results["planarity"] = tl.is_graph_planar(self.graph)
        results["connectedness"] = self.graph.is_connected()
        results["self_loops"] = tl.has_self_loops(self.graph)
        results["sparsity"] = tl.calculate_sparsity(self.graph)

        if self.graph.is_directed():
            results["local_clustering_coeffs"]  = self.graph.transitivity_local_directed()
            results["clustering_coeff"]  = self.graph.transitivity_avglocal_directed()      
        else:
            results["local_clustering_coeffs"] = self.graph.transitivity_local_undirected()
            results["clustering_coeff"] = self.graph.transitivity_undirected()
        
        results["nr_of_trinagles"] = len(self.graph.cliques(min=3, max=3))

        return results
    
