"""
Skeletonization and centerline extraction module for Banana Curve Analyzer.
Extracts the topological medial axis and traces the ordered tip-to-tip centerline.
"""

from typing import List, Tuple, Optional
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize


def extract_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    Compute the 1-pixel wide morphological skeleton of the binary mask.

    Args:
        binary_mask: Binary mask image (uint8, with values 0 and 255).

    Returns:
        np.ndarray: Skeleton image as uint8 (0 and 255).
    """
    bool_mask = binary_mask > 0
    if not np.any(bool_mask):
        return np.zeros_like(binary_mask, dtype=np.uint8)

    skel = skeletonize(bool_mask)
    return (skel.astype(np.uint8)) * 255


def build_skeleton_graph(skeleton: np.ndarray) -> nx.Graph:
    """
    Build a networkx graph from the 8-connected skeleton pixels.
    Edge weights represent Euclidean distance between adjacent pixels (1.0 or sqrt(2)).

    Args:
        skeleton: Skeleton image (uint8, 0 and 255).

    Returns:
        nx.Graph: Graph where nodes are (row, col) tuples.
    """
    g = nx.Graph()
    rows, cols = np.where(skeleton > 0)
    pixel_set = set(zip(rows, cols))

    # 8-neighborhood offsets and Euclidean step lengths
    neighbors = [
        (-1, -1, np.sqrt(2)), (-1, 0, 1.0), (-1, 1, np.sqrt(2)),
        (0, -1, 1.0),                        (0, 1, 1.0),
        (1, -1, np.sqrt(2)),  (1, 0, 1.0),  (1, 1, np.sqrt(2)),
    ]

    for r, c in pixel_set:
        g.add_node((r, c))
        for dr, dc, weight in neighbors:
            nr, nc = r + dr, c + dc
            if (nr, nc) in pixel_set:
                g.add_edge((r, c), (nr, nc), weight=weight)

    return g


def trace_longest_centerline(
    skeleton: np.ndarray,
) -> Tuple[List[Tuple[float, float]], Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """
    Find the longest geodesic path through the skeleton graph, effectively
    pruning minor spurs and selecting the true tip-to-tip banana centerline.

    Args:
        skeleton: Binary skeleton image (uint8).

    Returns:
        Tuple:
            - ordered_points: List of (x, y) coordinates in order from tip 1 to tip 2.
            - endpoint_1: (x, y) coordinates of the first endpoint.
            - endpoint_2: (x, y) coordinates of the second endpoint.
    """
    if cv2_count := np.count_nonzero(skeleton) < 2:
        return [], None, None

    g = build_skeleton_graph(skeleton)
    if len(g) < 2:
        return [], None, None

    # Focus on the largest connected component
    largest_cc = max(nx.connected_components(g), key=len)
    subg = g.subgraph(largest_cc).copy()

    # Find endpoints (degree == 1)
    endpoints = [node for node in subg.nodes() if subg.degree(node) == 1]

    best_path = []
    if len(endpoints) >= 2:
        # Find the pair of endpoints that has the longest shortest path
        max_dist = -1.0
        best_pair = (endpoints[0], endpoints[1])

        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                u, v = endpoints[i], endpoints[j]
                try:
                    dist = nx.shortest_path_length(subg, u, v, weight="weight")
                    if dist > max_dist:
                        max_dist = dist
                        best_pair = (u, v)
                except nx.NetworkXNoPath:
                    continue

        try:
            best_path = nx.shortest_path(subg, best_pair[0], best_pair[1], weight="weight")
        except nx.NetworkXNoPath:
            best_path = list(subg.nodes())
    elif len(endpoints) == 1:
        # Single endpoint (e.g. lasso or tail): find farthest node from that endpoint
        u = endpoints[0]
        lengths = nx.single_source_dijkstra_path_length(subg, u, weight="weight")
        farthest_node = max(lengths, key=lengths.get)
        best_path = nx.shortest_path(subg, u, farthest_node, weight="weight")
    else:
        # No degree 1 nodes (e.g. a loop): find two nodes with maximum distance
        nodes = list(subg.nodes())
        if len(nodes) >= 2:
            # Pick pair with maximum Euclidean distance as anchor
            coords = np.array(nodes)
            diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            sq_dists = np.sum(diffs ** 2, axis=-1)
            i, j = np.unravel_index(np.argmax(sq_dists), sq_dists.shape)
            u, v = nodes[i], nodes[j]
            try:
                best_path = nx.shortest_path(subg, u, v, weight="weight")
            except nx.NetworkXNoPath:
                best_path = nodes
        else:
            best_path = nodes

    if not best_path:
        return [], None, None

    # Convert nodes from (row, col) to standard Cartesian (x, y) = (col, row)
    ordered_points = [(float(c), float(r)) for (r, c) in best_path]
    endpoint_1 = ordered_points[0]
    endpoint_2 = ordered_points[-1]

    return ordered_points, endpoint_1, endpoint_2
