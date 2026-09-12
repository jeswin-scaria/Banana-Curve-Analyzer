"""
Skeletonization and centerline extraction module for Banana Curve Analyzer.
Extracts the topological medial axis and traces the ordered tip-to-tip centerline.
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2
import networkx as nx
from skimage.morphology import skeletonize

from .config import (
    SPUR_MIN_LENGTH_FLOOR_PX,
    SPUR_LENGTH_WIDTH_MULTIPLIER,
    SPUR_PRUNE_MAX_ITERATIONS,
)


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


def estimate_adaptive_spur_threshold(mask: np.ndarray, skeleton: np.ndarray) -> float:
    """
    Scale-adaptive minimum spur-branch length, in pixels.

    Reuses the mask's distance transform (the local half-width of the banana at
    each skeleton pixel) as a proxy for how "thick" this particular banana is in
    this particular image: a spur shorter than a small multiple of that width is
    almost certainly a boundary-noise artifact, not a real anatomical feature.
    """
    if np.count_nonzero(skeleton) == 0:
        return SPUR_MIN_LENGTH_FLOOR_PX

    dist_transform = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    widths = dist_transform[skeleton > 0]
    if widths.size == 0:
        return SPUR_MIN_LENGTH_FLOOR_PX

    median_half_width = float(np.median(widths))
    return max(SPUR_MIN_LENGTH_FLOOR_PX, SPUR_LENGTH_WIDTH_MULTIPLIER * median_half_width)


def _trace_branch_from_endpoint(
    g: nx.Graph,
    endpoint: Tuple[int, int],
) -> Tuple[List[Tuple[int, int]], float, Tuple[int, int]]:
    """Walk from a degree-1 node until hitting a branch point (degree != 2) or
    another endpoint. Returns (path_nodes, path_length, terminating_node)."""
    path = [endpoint]
    length = 0.0
    prev: Optional[Tuple[int, int]] = None
    current = endpoint

    while True:
        neighbors = [n for n in g.neighbors(current) if n != prev]
        if not neighbors:
            return path, length, current
        nxt = neighbors[0]
        length += float(g[current][nxt]["weight"])
        path.append(nxt)
        if g.degree(nxt) != 2:
            return path, length, nxt
        prev, current = current, nxt


def prune_skeleton_spurs(
    skeleton: np.ndarray,
    min_branch_length: float,
    max_iterations: int = SPUR_PRUNE_MAX_ITERATIONS,
) -> np.ndarray:
    """
    Remove short degree-1 branches ("spurs") from a skeleton before centerline
    tracing. A banana skeleton should topologically be a single unbranched path;
    any branch point is presumptively noise from boundary irregularities, and
    the short dead-end hanging off it is pruned rather than left to compete as
    a candidate endpoint.

    Endpoints of a skeleton with no branch points at all (already a simple path)
    are left untouched -- pruning only removes a branch that terminates at a
    genuine branch point (degree >= 3), never a shape's real two tips.
    """
    skel = skeleton.copy()

    for _ in range(max_iterations):
        g = build_skeleton_graph(skel)
        if len(g) == 0:
            break

        endpoints = [n for n in g.nodes() if g.degree(n) == 1]
        if len(endpoints) <= 2:
            break

        nodes_to_remove = set()
        for ep in endpoints:
            if ep in nodes_to_remove:
                continue
            path, length, end_node = _trace_branch_from_endpoint(g, ep)
            end_degree = g.degree(end_node) if end_node in g else 0
            if length < min_branch_length and end_degree >= 3:
                nodes_to_remove.update(path[:-1])  # keep the branch point itself

        if not nodes_to_remove:
            break

        for (r, c) in nodes_to_remove:
            skel[r, c] = 0

    return skel


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
    if np.count_nonzero(skeleton) < 2:
        return [], None, None

    g = build_skeleton_graph(skeleton)
    if len(g) < 2:
        return [], None, None

    # Focus on the largest connected component
    largest_cc = max(nx.connected_components(g), key=len)
    subg = g.subgraph(largest_cc).copy()

    # Find the geodesic diameter path (longest shortest path through the skeleton).
    # The 2-sweep Dijkstra algorithm is mathematically guaranteed to find the true
    # extreme geometric tips of the tree/skeleton, avoiding failure modes where thinning
    # leaves a small 2- or 3-pixel clique at a tip (so degree != 1).
    start_node = next(iter(subg.nodes()))
    lengths1 = nx.single_source_dijkstra_path_length(subg, start_node, weight="weight")
    u = max(lengths1, key=lengths1.get)

    lengths2 = nx.single_source_dijkstra_path_length(subg, u, weight="weight")
    v = max(lengths2, key=lengths2.get)

    # If degree-1 endpoints exist, also check them to ensure maximal coverage
    endpoints = [node for node in subg.nodes() if subg.degree(node) == 1]
    best_dist = lengths2.get(v, 0.0)
    best_pair = (u, v)

    for ep in endpoints:
        if ep in lengths2 and lengths2[ep] > best_dist:
            best_dist = lengths2[ep]
            best_pair = (u, ep)

    try:
        best_path = nx.shortest_path(subg, best_pair[0], best_pair[1], weight="weight")
    except nx.NetworkXNoPath:
        best_path = list(subg.nodes())

    if not best_path:
        return [], None, None

    # Convert nodes from (row, col) to standard Cartesian (x, y) = (col, row)
    ordered_points = [(float(c), float(r)) for (r, c) in best_path]
    endpoint_1 = ordered_points[0]
    endpoint_2 = ordered_points[-1]

    return ordered_points, endpoint_1, endpoint_2


def extend_centerline_to_boundary(
    ordered_points: List[Tuple[float, float]],
    mask: np.ndarray,
    max_search_dist: int = 150,
) -> List[Tuple[float, float]]:
    """
    Extend the centerline points at both ends to reach the physical boundary
    of the banana mask. Thinning/skeletonization inherently stops roughly one
    radius short of the shape's boundary at each tip. This function shoots rays
    outward along the end-tangent directions and extends the centerline to the
    outermost boundary of the mask, ensuring true end-to-end banana coverage.
    """
    if not ordered_points or len(ordered_points) < 2 or mask is None:
        return ordered_points

    h, w = mask.shape[:2]
    pts_arr = np.array(ordered_points, dtype=np.float64)

    def _ray_to_boundary(at_start: bool) -> List[Tuple[float, float]]:
        k = min(10, len(pts_arr) - 1)
        if at_start:
            direction = pts_arr[0] - pts_arr[k]
            p_end = pts_arr[0]
        else:
            direction = pts_arr[-1] - pts_arr[-1 - k]
            p_end = pts_arr[-1]

        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return []

        u = direction / norm
        last_inside = p_end.copy()
        ext_pts: List[Tuple[float, float]] = []

        limit = min(max_search_dist, max(h, w) // 3)
        for step in range(1, limit):
            test_pt = p_end + u * step
            ix, iy = int(round(test_pt[0])), int(round(test_pt[1]))
            if 0 <= iy < h and 0 <= ix < w:
                if mask[iy, ix] > 0:
                    last_inside = test_pt
                    if step % 2 == 0:
                        ext_pts.append((float(test_pt[0]), float(test_pt[1])))
                else:
                    break
            else:
                break

        last_pt = (float(last_inside[0]), float(last_inside[1]))
        if len(ext_pts) == 0 and not np.allclose(last_inside, p_end):
            ext_pts = [last_pt]
        elif len(ext_pts) > 0 and not np.allclose(ext_pts[-1], last_inside):
            ext_pts.append(last_pt)

        return ext_pts

    ext_start = _ray_to_boundary(at_start=True)
    ext_end = _ray_to_boundary(at_start=False)

    result = list(ordered_points)
    if ext_start:
        result = ext_start[::-1] + result
    if ext_end:
        result = result + ext_end

    return result

