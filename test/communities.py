import igraph as ig

def detect_communities(graph):
    """
    Apply various community detection algorithms on a graph and print results.

    Parameters:
        graph (igraph.Graph): The input graph for community detection.

    Returns:
        None
    """
    print("Edge Betweenness:")
    dendrogram = graph.community_edge_betweenness(directed=False, weights=None)
    clusters = dendrogram.as_clustering()
    print(clusters.membership)

    print("\nWalktrap:")
    walktrap = graph.community_walktrap(weights=None, steps=4)
    clusters = walktrap.as_clustering()
    print(clusters.membership)

    print("\nInfomap:")
    clusters = graph.community_infomap(edge_weights=None, vertex_weights=None, trials=10)
    print(clusters.membership)

    print("\nLabel Propagation:")
    clusters = graph.community_label_propagation(weights=None, initial=None, fixed=None)
    print(clusters.membership)

    print("\nLeading Eigenvector:")
    clusters = graph.community_leading_eigenvector(weights=None, clusters=None, arpack_options=None)
    print(clusters.membership)

    print("\nMultilevel (Louvain):")
    clusters = graph.community_multilevel(weights=None, resolution=1.0, return_levels=False)
    print(clusters.membership)

    print("\nFast Greedy:")
    dendrogram = graph.community_fastgreedy(weights=None)
    clusters = dendrogram.as_clustering()
    print(clusters.membership)

    print("\nSpinglass:")
    try:
        clusters = graph.community_spinglass(weights=None, spins=25, parupdate=False, start_temp=1.0, stop_temp=0.01, gamma=1.0, update_rule="simple")
        print(clusters.membership)
    except Exception as e:
        print(f"Spinglass failed: {e}")

    print("\nOptimal Modularity:")
    try:
        clusters = graph.community_optimal_modularity(weights=None)
        print(clusters.membership)
    except Exception as e:
        print(f"Optimal Modularity failed: {e}")

if __name__ == "__main__":
    # Create a sample graph (Zachary's Karate Club)
    g = ig.Graph.Famous("Zachary")

    # Run community detection algorithms
    detect_communities(g)

