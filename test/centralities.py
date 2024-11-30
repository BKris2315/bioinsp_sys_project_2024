import igraph as ig

def calculate_centrality_measures(graph):
    """
    Calculate and print all centrality measures implemented in igraph.

    Parameters:
        graph (igraph.Graph): The input graph for centrality calculation.

    Returns:
        dict: A dictionary containing centrality measures for all nodes.
    """
    centrality_measures = {}

    # Degree Centrality
    print("Calculating Degree Centrality...")
    centrality_measures['degree'] = graph.degree()

    # Closeness Centrality
    print("Calculating Closeness Centrality...")
    centrality_measures['closeness'] = graph.closeness(normalized=True)

    # Betweenness Centrality
    print("Calculating Betweenness Centrality...")
    centrality_measures['betweenness'] = graph.betweenness(vertices=None, directed=False, weights=None, cutoff=None)

    # Eigenvector Centrality
    print("Calculating Eigenvector Centrality...")
    centrality_measures['eigenvector'] = graph.eigenvector_centrality(directed=False, scale=True, weights=None, return_eigenvalue=False)

    # PageRank
    print("Calculating PageRank...")
    centrality_measures['pagerank'] = graph.pagerank(directed=False, damping=0.85, weights=None, arpack_options=None)

    # Harmonic Centrality
    print("Calculating Harmonic Centrality...")
    centrality_measures['harmonic'] = graph.harmonic_centrality(vertices=None, mode='ALL', weights=None)

    # Load Centrality
    print("Calculating Load Centrality...")
    centrality_measures['load'] = calculate_load_centrality(graph)

    # Katz Centrality
    print("Calculating Katz Centrality...")
    centrality_measures['katz'] = calculate_katz_centrality(graph)

    return centrality_measures

def calculate_load_centrality(graph):
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

def calculate_katz_centrality(graph, alpha=0.1, beta=1.0, tol=1e-9, maxiter=1000):
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

if __name__ == "__main__":
    # Create a sample graph (Zachary's Karate Club)
    g = ig.Graph.Famous("Zachary")

    # Calculate centrality measures
    centrality_results = calculate_centrality_measures(g)

    # Print results
    for measure, values in centrality_results.items():
        print(f"\n{measure.capitalize()} Centrality:")
        print(values)
