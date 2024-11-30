The `igraph` library in Python provides several methods for finding communities in graphs. These methods are based on different algorithms and are suitable for various types of community detection tasks.

Here is a comprehensive list of the community detection methods in `igraph`, along with examples of their usage:

---

### **1. Edge Betweenness**
- **Description**: Detects communities by progressively removing edges with the highest betweenness.
- **Method**: `graph.community_edge_betweenness()`
- **Example**:
    ```python
    import igraph as ig

    g = ig.Graph.Famous("Zachary")  # Example graph: Zachary's Karate Club
    dendrogram = g.community_edge_betweenness()
    clusters = dendrogram.as_clustering()
    print(clusters.membership)  # Community membership of nodes
    ```

---

### **2. Walktrap**
- **Description**: Detects communities using random walks. Nodes that are visited together during walks are grouped into the same community.
- **Method**: `graph.community_walktrap()`
- **Example**:
    ```python
    walktrap = g.community_walktrap()
    clusters = walktrap.as_clustering()
    print(clusters.membership)
    ```

---

### **3. Infomap**
- **Description**: Detects communities by minimizing the map equation, which captures the flow of information in a network.
- **Method**: `graph.community_infomap()`
- **Example**:
    ```python
    clusters = g.community_infomap()
    print(clusters.membership)
    ```

---

### **4. Label Propagation**
- **Description**: Detects communities by propagating labels through the graph.
- **Method**: `graph.community_label_propagation()`
- **Example**:
    ```python
    clusters = g.community_label_propagation()
    print(clusters.membership)
    ```

---

### **5. Leading Eigenvector**
- **Description**: Detects communities by splitting the graph using the leading eigenvector of the modularity matrix.
- **Method**: `graph.community_leading_eigenvector()`
- **Example**:
    ```python
    clusters = g.community_leading_eigenvector()
    print(clusters.membership)
    ```

---

### **6. Multilevel (Louvain)**
- **Description**: Uses a multilevel optimization of modularity to find communities. This is often referred to as the Louvain method.
- **Method**: `graph.community_multilevel()`
- **Example**:
    ```python
    clusters = g.community_multilevel()
    print(clusters.membership)
    ```

---

### **7. Fast Greedy**
- **Description**: Detects communities by greedily optimizing modularity.
- **Method**: `graph.community_fastgreedy()`
- **Example**:
    ```python
    dendrogram = g.community_fastgreedy()
    clusters = dendrogram.as_clustering()
    print(clusters.membership)
    ```

---

### **8. Spinglass**
- **Description**: Based on a spin-glass model, suitable for weighted or unweighted graphs. Works only on connected graphs.
- **Method**: `graph.community_spinglass()`
- **Example**:
    ```python
    clusters = g.community_spinglass()
    print(clusters.membership)
    ```

---

### **9. Optimal Modularity**
- **Description**: Finds the partition that maximizes modularity. This method is computationally expensive.
- **Method**: `graph.community_optimal_modularity()`
- **Example**:
    ```python
    clusters = g.community_optimal_modularity()
    print(clusters.membership)
    ```

---

### **10. Hierarchical Random Walks**
- **Description**: Similar to Walktrap but based on hierarchical random walks.
- **Method**: `graph.community_walktrap()`
- **Example**:
    ```python
    walktrap = g.community_walktrap()
    clusters = walktrap.as_clustering()
    print(clusters.membership)
    ```

---

### **11. Community Detection on Directed Graphs**
- **Description**: Some methods like Infomap also support directed graphs.
- **Example**:
    ```python
    directed_graph = ig.Graph.Directed.Famous("Zachary")
    clusters = directed_graph.community_infomap()
    print(clusters.membership)
    ```

---

### **Comparison of Methods**

| Method                  | Weighted Graphs | Directed Graphs | Handles Large Graphs | Optimizes Modularity | Notes                         |
|-------------------------|-----------------|-----------------|----------------------|----------------------|-------------------------------|
| Edge Betweenness        | Yes             | No              | No                   | No                   | Computationally expensive.   |
| Walktrap                | Yes             | No              | Yes                  | Yes                  | Effective for large graphs.  |
| Infomap                 | Yes             | Yes             | Yes                  | No                   | Based on information flow.   |
| Label Propagation       | No              | No              | Yes                  | No                   | Fast but non-deterministic.  |
| Leading Eigenvector     | No              | No              | No                   | Yes                  | Best for smaller graphs.     |
| Multilevel (Louvain)    | Yes             | No              | Yes                  | Yes                  | Scales well; Louvain method. |
| Fast Greedy             | Yes             | No              | Yes                  | Yes                  | Fast and efficient.          |
| Spinglass               | Yes             | No              | No                   | No                   | Works for weighted graphs.   |
| Optimal Modularity      | Yes             | No              | No                   | Yes                  | Computationally expensive.   |

---

### **Choosing a Method**
- For large graphs, use **Multilevel (Louvain)** or **Label Propagation**.
- For directed graphs, use **Infomap**.
- For modularity optimization, use **Fast Greedy** or **Multilevel (Louvain)**.

Let me know if you need more details or help implementing any specific method!