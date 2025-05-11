from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import networkx as nx
import numpy as np
import shapely
import sknw
from anndata import AnnData
from matplotlib.path import Path
from rasterio import features
from scipy.spatial import Delaunay
from scipy.optimize import root_scalar
from shapely import geometry
from shapely.ops import polygonize, unary_union
from skimage.morphology import skeletonize
from squidpy._docs import d
import sys
import alphashape
import trimesh
import warnings

def _alphashape_optimizealpha(
    points: np.ndarray,
    component: int,
    max_iterations: int = 10000,
    lower: float | None = None,
    upper: float | None = None,
    silent: bool = False
):
    """
    Wrapper function for alphashape.optimizealpha.
    """
    dim = points.shape[1]
    if points.shape[0] < (dim+1):
        if not silent:
            warnings.warn(f'Component {component}: not enough points to form a shape.')
        return component, np.nan

    # Set default bounds (as 1/bound to agree with alphashape definition of alpha)
    if upper is None:
        upper = sys.float_info.max
    else:
        upper = 1 / upper
    if lower is None:
        if dim == 2:
            lower = 0.
        elif dim == 3:
            # If zero, alphashape function would return polygon instead of mesh
            lower = sys.float_info.min
    else:
        lower = 1 / lower


    # Run optimization
    alpha_inv = alphashape.optimizealpha(points, max_iterations=max_iterations, lower=lower, upper=upper, silent=silent)

    if dim == 2:
        return component, alpha_inv
    if dim == 3:
        if alpha_inv == 0:
            if not silent:
                warnings.warn(f'Component {component}: no shape found, '
                            'alpha range might be too stringent.')
            return component, np.nan
        else:
            return component, alpha_inv

@d.dedent
def alphashape_optimize(
    adata: AnnData,
    cluster_key: str = 'component',
    max_iterations: int = 10000,
    lower: float | None = None,
    upper: float | None = None,
    silent: bool = False,
) -> tuple[ dict[int, float] , dict[int, trimesh.Trimesh] ] | tuple[ dict[int, float] , dict[int, geometry.Polygon] ]:
    """
    Function to optimize alpha using alphashape.optimizealpha.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the component labels are stored.
    max_iterations
        Maximum number of iterations for the alpha shape optimizer.
    lower
        Minimum value for the alpha parameter of the alpha shape algorithm. If not specified, set to lowest possible value.
    upper
        Maximum value for the alpha parameter of the alpha shape algorithm. If not specified, set to highest possible value.
    silent
        If ``True``, suppresses function-specific warnings.

    %(copy)s
    Returns
    -------
    Returns a tuple with two :class:`dict`, each with the component labels as keys.
    In the first dictionary, values are the optimal alpha, in the second dictionary, values are the corresponding shape (boundary).
    If no shape is found, np.nan and None are returned for alpha and shape, respectively.
    """
    assert (adata.obsm["spatial"].shape[1] == 2) | (adata.obsm["spatial"].shape[1] == 3), "Points must be 2D or 3D."

    components = [component for component in adata.obs[cluster_key].unique() if component != -1 and not np.isnan(component)]

    alphas = {}
    shapes = {}
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                _alphashape_optimizealpha,
                adata.obsm["spatial"][adata.obs[cluster_key] == component, :],
                component,
                max_iterations,
                lower,
                upper,
                silent
            ): component
            for component in components
        }

        for future in as_completed(futures):
            component, alpha_inv = future.result()
            alphas[component] = 1/alpha_inv
            if not np.isnan(alpha_inv):
                shapes[component] = alphashape.alphashape(adata.obsm["spatial"][adata.obs[cluster_key] == component, :], alpha_inv)
            else:
                shapes[component] = None

    return alphas, shapes

def _alpha_shape_2D(coords, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.

    Adapted from `here <https://web.archive.org/web/20200726174718/http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/>`_.

    Parameters
    ----------
    coords : np.array
        Array of coordinates of points.
    alpha : float
        Alpha value to influence the gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers. Too large, and you lose
        everything!
    Returns
    -------
    concave_hull : shapely.geometry.Polygon
        Concave hull of the points.
    """
    tri = Delaunay(coords)
    triangles = coords[tri.simplices]
    a = ((triangles[:, 0, 0] - triangles[:, 1, 0]) ** 2 + (triangles[:, 0, 1] - triangles[:, 1, 1]) ** 2) ** 0.5
    b = ((triangles[:, 1, 0] - triangles[:, 2, 0]) ** 2 + (triangles[:, 1, 1] - triangles[:, 2, 1]) ** 2) ** 0.5
    c = ((triangles[:, 2, 0] - triangles[:, 0, 0]) ** 2 + (triangles[:, 2, 1] - triangles[:, 0, 1]) ** 2) ** 0.5
    s = (a + b + c) / 2.0
    areas = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < alpha]
    edge1 = filtered[:, (0, 1)]
    edge2 = filtered[:, (1, 2)]
    edge3 = filtered[:, (2, 0)]
    edge_points = np.unique(np.concatenate((edge1, edge2, edge3)), axis=0)  # .tolist()
    m = geometry.MultiLineString(edge_points.tolist())
    triangles = list(polygonize(m.geoms))
    return unary_union(triangles), triangles, edge_points

def _mesh_is_valid(
    mesh: trimesh.Trimesh,
    points: np.ndarray
    ) -> bool:
    return len(mesh.faces) > 0 and mesh.is_watertight and all(
        trimesh.proximity.signed_distance(mesh, list(points)) >= 0 )

def _process_component(points, component, hole_area_ratio=0.1, alpha_start=None):
    """
    Guide the alpha shape creation.
    For 2D data, alpha_start is doubled until a valid shape is found
    (min. 10 edge points, and small holes integrated based on hole_area_ratio).
    For 3D data, if alpha_start does not provide a valid shape,
    alpha is optimized using alphashape.optimizealpha.
    Note that the alpha value used by the alphashape package seems to be the
    inverse of the alpha value used by cellcharter. For clarity, we use the cellcharter
    definition and pass the inverse to alphashape functions.
    """
    if points.shape[1] == 2:
        if alpha_start is not None:
            alpha = alpha_start
        else:
            alpha = 2000
        polygon, triangles, edge_points = _alpha_shape_2D(points, alpha)

        while (
            type(polygon) is not geometry.polygon.Polygon
            or type(polygon) is geometry.MultiPolygon
            or edge_points.shape[0] < 10
        ):
            alpha *= 2
            polygon, triangles, edge_points = _alpha_shape_2D(points, alpha)

        boundary_with_holes = max(triangles, key=lambda triangle: triangle.area)
        boundary = polygon

        for interior in boundary_with_holes.interiors:
            interior_polygon = geometry.Polygon(interior)
            hole_to_boundary_ratio = interior_polygon.area / boundary.area
            if hole_to_boundary_ratio > hole_area_ratio:
                try:
                    difference = boundary.difference(interior_polygon)
                    if isinstance(difference, geometry.Polygon):
                        boundary = difference
                except Exception:  # noqa: B902
                    pass
        return component, boundary

    else: # 3D
        if alpha_start is None:
            alpha_start = sys.float_info.min # must be > 0
        alpha = 1 / alpha_start

        # TODO: what about shapes with enclosed holes? (relevant for purity etc)

        # Check if initial alpha returns valid shape
        mesh = alphashape.alphashape(points, alpha)
        if isinstance(mesh, trimesh.base.Trimesh):
            if _mesh_is_valid(mesh, points):
                return component, mesh

        # Otherwise, optimize alpha
        # Using manual optimization, which is faster than alphashape.optimizealpha
        # but might result in a slightly larger alpha
        factor_increase = 2
        factor_decrease = .75
        n_iter = 10
        # Increase alpha until valid shape is found
        while not _mesh_is_valid(mesh, points):
            alpha *= factor_increase
            mesh = alphashape.alphashape(points, alpha=1/alpha)
        # Decrease alpha until invalid shape is found
        alpha0 = alpha
        mesh0 = mesh.copy()
        for i in range(n_iter):
            alpha1 = alpha0 * factor_decrease
            mesh1 = alphashape.alphashape(points, alpha=1/alpha1)
            if not _mesh_is_valid(mesh1, points): ## invalid, return previous mesh
                return component, mesh0
            else: ## still valid, could be reduced more
                alpha0 = alpha1
                mesh0 = mesh1.copy()
        return component, mesh0 ## return the last valid mesh

@d.dedent
def boundaries(
    adata: AnnData,
    cluster_key: str = "component",
    min_hole_area_ratio: float = 0.1,
    alpha_start: int = 2000,
    copy: bool = False,
) -> None | dict[int, geometry.Polygon] | dict[int, trimesh.Trimesh]:
    """
    Compute the topological boundaries of sets of cells.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    min_hole_area_ratio
        Minimum ratio between the area of a hole and the area of the boundary (only relevant for 2D data).
    alpha_start
        Starting value for the alpha parameter of the alpha shape algorithm.

    %(copy)s
    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the boundaries as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['boundaries']`` - the above mentioned :class:`dict`.
    """
    assert 0 <= min_hole_area_ratio <= 1, "min_hole_area_ratio must be between 0 and 1"
    assert alpha_start > 0, "alpha_start must be greater than 0"
    assert (adata.obsm["spatial"].shape[1] == 2) | (adata.obsm["spatial"].shape[1] == 3), "adata.obsm['spatial'] must be of shape Nx2 or Nx3"

    clusters = [cluster for cluster in adata.obs[cluster_key].unique() if cluster != -1 and not np.isnan(cluster)]

    boundaries = {}
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                _process_component,
                adata.obsm["spatial"][adata.obs[cluster_key] == cluster, :],
                cluster,
                min_hole_area_ratio,
                alpha_start,
            ): cluster
            for cluster in clusters
        }

        for future in as_completed(futures):
            component, boundary = future.result()
            boundaries[component] = boundary

    if copy:
        return boundaries

    adata.uns[f"shape_{cluster_key}"] = {"boundary": boundaries}

def _find_longest_simple_path(graph: nx.Graph) -> list:
    """Find the longest simple path in a graph using DFS (brute-force)."""
    longest_path = []
    longest_length = 0
    weight = nx.get_edge_attributes(graph, "weight")

    def dfs(node, visited, path, length):
        nonlocal longest_path, longest_length
        visited.add(node)
        path.append(node)
        if length > longest_length:
            longest_length = length
            longest_path = list(path)

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                edge = (min(node, neighbor), max(node, neighbor))
                dfs(neighbor, visited, path, length + weight.get(edge, 1))
        path.pop()
        visited.remove(node)

    for start_node in graph.nodes:
        dfs(start_node, set(), [], 0)

    return longest_path, longest_length

def _prune_graph(
        graph: nx.Graph,
        total_length:float,
        longest_path_nodes:set,
        min_ratio=0.05
    ) -> nx.Graph:

    nodes_to_remove = []
    for node in graph.nodes:
        if node not in longest_path_nodes:
            # Consider node's total edge weight
            edge_weights = [
                graph.edges[(node, nbr)]['weight']
                for nbr in graph.neighbors(node)
                if (node, nbr) in graph.edges or (nbr, node) in graph.edges
            ]
            node_total_weight = np.sum(edge_weights)
            if node_total_weight < min_ratio * total_length:
                nodes_to_remove.append(node)

    pruned_graph = graph.copy()
    pruned_graph.remove_nodes_from(nodes_to_remove)
    return pruned_graph

def _linearity(boundary, dim, height, min_ratio=0.05):
    # Voxelize or rasterize the boundary
    if dim == 2: # Scale and rasterize
        img, _ = _rasterize(boundary, height=height)
    else: # Turn into voxel grid, scale by setting voxel side length
        img = _voxelize(boundary, height=height)

    # Skeletonize & encode skeleton as graph
    skeleton = skeletonize(img).astype(int) ## will use method='lee' for 3D by default
    graph = sknw.build_sknw(skeleton.astype(np.uint16))
    graph = graph.to_undirected()

    # Find longest path (acyclic)
    longest_path, _ = _find_longest_simple_path(graph)
    longest_path_nodes = set(longest_path)

    # Prune 'spurious' branches while protecting the longest path
    # Note: This will not remove branches that are part of cycles
    total_length = np.sum(list(nx.get_edge_attributes(graph, "weight").values())) # unpruned
    pruned_graph = _prune_graph(graph, total_length, longest_path_nodes, min_ratio=min_ratio)

    # Compute the linearity score: longest path length / total length of the pruned graph
    path_length = nx.path_weight(pruned_graph, longest_path, weight="weight")
    total_length = np.sum(list(nx.get_edge_attributes(pruned_graph, "weight").values())) # pruned

    return ( path_length / total_length if total_length > 0 else 0. )


def _rasterize(boundary, height=1000):
    minx, miny, maxx, maxy = boundary.bounds
    poly = shapely.affinity.translate(boundary, -minx, -miny)
    if maxx - minx > maxy - miny:
        scale_factor = height / poly.bounds[2]
    else:
        scale_factor = height / poly.bounds[3]
    poly = shapely.affinity.scale(poly, scale_factor, scale_factor, origin=(0, 0, 0))
    return features.rasterize([poly], out_shape=(height, int(height * (maxx - minx) / (maxy - miny)))), scale_factor

def _voxelize(boundary, height=100):
    # TODO tests
    bbox_sides = boundary.bounding_box.extents # axis-aligned, as voxels will be, too
    voxel_length = bbox_sides.max() / height
    voxelized = boundary.voxelized(voxel_length)
    voxelized = voxelized.fill(method='holes')

    return voxelized.matrix*1

@d.dedent
def linearity(
    adata: AnnData,
    cluster_key: str = "component",
    out_key: str = "linearity",
    height: int | None = None,
    min_ratio: float = 0.05,
    copy: bool = False,
) -> None | dict[int, float]:
    """
    Compute the linearity of the topological boundaries of sets of cells.

    It rasterizes the polygon and computes the skeleton of the rasterized image.
    Then, it computes the longest path in the skeleton and divides it by the total length of the skeleton.
    Branches that are shorter than ``min_ratio`` times the total length of the skeleton are removed because are not considered real branches.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    out_key
        Key in :attr:`anndata.AnnData.obs` where the metric values are stored if ``copy = False``.
    height
        Height of the rasterized image. The lengths of the other side(s) is/are computed automatically to preserve the aspect ratio of the shape. Higher values lead to more precise results but also higher memory usage. Default for 2D data is 1000, for 3D data 100.
    min_ratio
        Minimum ratio between the length of a branch and the total length of the skeleton to be considered a real branch and not be removed.
    %(copy)s
    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the linearity as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['{{out_key}}']`` - - the above mentioned :class:`dict`.
    """
    boundaries = adata.uns[f"shape_{cluster_key}"]["boundary"]

    # Determine default height from type of boundary
    # Cave: assuming all input boundary objects are of the same dimensionality
    if isinstance(boundaries[0], geometry.Polygon) | isinstance(boundaries[0], geometry.MultiPolygon):
        dim = 2
        height = 1000
    elif isinstance(boundaries[0], trimesh.Trimesh):
        dim = 3
        height = 100
    else:
        raise ValueError("Unknown boundary type. Must be either 2D or 3D.")

    linearity_score = {}
    for cluster, boundary in boundaries.items():
        linearity_score[cluster] = _linearity(boundary, dim=dim, height=height, min_ratio=min_ratio)

    if copy:
        return linearity_score

    adata.uns[f"shape_{cluster_key}"][out_key] = linearity_score


def _elongation(boundary):
    if isinstance(boundary, geometry.Polygon) | isinstance(boundary, geometry.MultiPolygon): ## 2D
        # get the minimum bounding rectangle and zip coordinates into a list of point-tuples
        mbr_points = list(zip(*boundary.minimum_rotated_rectangle.exterior.coords.xy))

        # calculate the length of each side of the minimum bounding rectangle
        mbr_lengths = [geometry.LineString((mbr_points[i], mbr_points[i + 1])).length for i in range(len(mbr_points) - 1)]

        # get major/minor axis measurements
        minor_axis = min(mbr_lengths)
        major_axis = max(mbr_lengths)
        return 1 - minor_axis / major_axis

    else: ## 3D
        bbox = boundary.bounding_box_oriented
        sides = np.array( bbox.primitive.extents )
        sides.sort() # ascending
        return 1 - (sides[1] / sides[2]) ## second longest / longest

@d.dedent
def elongation(
    adata: AnnData,
    cluster_key: str = "component",
    out_key: str = "elongation",
    copy: bool = False,
) -> None | dict[int, float]:
    """
    Compute the elongation of the topological boundaries of sets of cells.

    It computes the minimum bounding rectangle (in 3D: box) of the boundary and divides the length of the minor axis by the length of the major axis
    or, in 3D, the second longest side by the longest side.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    out_key
        Key in :attr:`anndata.AnnData.obs` where the metric values are stored if ``copy = False``.
    %(copy)s
    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the elongation as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['{{out_key}}']`` - - the above mentioned :class:`dict`.
    """
    boundaries = adata.uns[f"shape_{cluster_key}"]["boundary"]

    elongation_score = {}
    for cluster, boundary in boundaries.items():
        elongation_score[cluster] = _elongation(boundary)

    if copy:
        return elongation_score
    adata.uns[f"shape_{cluster_key}"][out_key] = elongation_score

def _flatness(mesh):
    if not isinstance(mesh, trimesh.Trimesh):
        return None
    bbox = mesh.bounding_box_oriented # TODO check for valid box
    sides = np.array( bbox.primitive.extents )
    sides.sort() # ascending
    return 1 - (sides[0] / sides[1]) ## shortest / second longest

@d.dedent
def flatness(
    adata: AnnData,
    cluster_key: str = "component",
    out_key: str = "flatness",
    copy: bool = False,
) -> None | dict[int, float]:
    """
    Compute the flatness of the 3D topological boundaries of sets of cells.

    It computes the oriented minimum bounding box of the boundary and divides the length of the shortest side by the length of the second longest side.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    out_key
        Key in :attr:`anndata.AnnData.obs` where the metric values are stored if ``copy = False``.
    %(copy)s
    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the flatness as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['{{out_key}}']`` - - the above mentioned :class:`dict`.
    """
    boundaries = adata.uns[f"shape_{cluster_key}"]["boundary"]

    flatness_score = {}
    for cluster, boundary in boundaries.items():
        flatness_score[cluster] = _flatness(boundary)

    if copy:
        return flatness_score
    adata.uns[f"shape_{cluster_key}"][out_key] = flatness_score

def _sphericity(mesh):
    if not isinstance(mesh, trimesh.Trimesh):
        return None
    if not mesh.is_watertight:
        return None
    volume = mesh.volume
    area = mesh.area
    if area == 0:
        return None
    return ( (np.pi**(1/3) * ((6*volume)**(2/3))) / area )

@d.dedent
def sphericity(
    adata: AnnData,
    cluster_key: str = "component",
    out_key: str = "sphericity",
    copy: bool = False,
) -> None | dict[int, float]:
    """
    Compute the sphericity of the 3D topological boundaries of sets of cells.

    Sphericity is computed following the definition by Wadell (1935), which uses the area and volume of the alpha shape.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    out_key
        Key in :attr:`anndata.AnnData.obs` where the metric values are stored if ``copy = False``.
    %(copy)s
    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the sphericity as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['{{out_key}}']`` - - the above mentioned :class:`dict`.
    """
    boundaries = adata.uns[f"shape_{cluster_key}"]["boundary"]

    sphericity_score = {}
    for cluster, boundary in boundaries.items():
        sphericity_score[cluster] = _sphericity(boundary)

    if copy:
        return sphericity_score
    adata.uns[f"shape_{cluster_key}"][out_key] = sphericity_score


def _axes(boundary):
    # get the minimum bounding rectangle and zip coordinates into a list of point-tuples
    mbr_points = list(zip(*boundary.minimum_rotated_rectangle.exterior.coords.xy))
    # calculate the length of each side of the minimum bounding rectangle
    mbr_lengths = [geometry.LineString((mbr_points[i], mbr_points[i + 1])).length for i in range(len(mbr_points) - 1)]
    return min(mbr_lengths), max(mbr_lengths)

def _major_axis(boundary: trimesh.Trimesh) -> float:
    # PCA on vertices to get principal axes
    points = boundary.vertices - boundary.centroid # center points
    eigvals, eigvecs = np.linalg.eigh(np.cov(points.T)) # correspond to principal axes
    axis = eigvecs[:, np.argmax(eigvals)]
    # Project points onto the major axis
    projections = points @ axis
    return projections.max() - projections.min()

def _fiber_length_cylinder(volume, area):
    # Solve for length L in a cylinder
    # with same volume V and surface area S:
    # V = πr^2 L
    # S = 2πr^2 + 2πrL
    def f(L):
        if L <= 0:
            return np.inf
        r2 = volume / (np.pi * L)
        r = np.sqrt(r2)
        surface = 2 * np.pi * r2 + 2 * np.pi * r * L
        return surface - area

    # Use a root-finding method to find L
    # Function must change sign within the bracketing interval
    # Try a range of logarithmic values for L
    # to find a root
    L_range = np.logspace(-3, 6, 100)
    values = [f(L) for L in L_range]
    for i in range(len(values) - 1):
        if values[i] * values[i + 1] < 0:
            result = root_scalar(f, bracket=[L_range[i], L_range[i+1]], method='brentq')
            if result.converged:
                return result.root
    return None # failed to solve for L, likely shape too dissimilar from cylinder

def _fiber_length_cuboid(volume, area):
    # Assume square cross-section: w² * L = V, S = 2w² + 4wL
    def f(L):
        if L <= 0:
            return np.inf
        w2 = volume / L
        w = np.sqrt(w2)
        surface = 2 * w2 + 4 * w * L
        return surface - area

    # Find L (as for cylinder)
    L_range = np.logspace(-3, 6, 100)
    values = [f(L) for L in L_range]
    for i in range(len(values) - 1):
        if values[i] * values[i + 1] < 0:
            result = root_scalar(f, bracket=[L_range[i], L_range[i+1]], method='brentq')
            if result.converged:
                return result.root
    return None # failed to solve for L, likely shape too dissimilar from cuboid

def _curl_3D_shape_comparison(mesh: trimesh.Trimesh) -> float:
    if not mesh.is_watertight or mesh.volume < 1e-6 or mesh.area < 1e-6:
        return None

    V = mesh.volume
    A = mesh.area
    L_major = _major_axis(mesh)

    # Try cylinder
    L_fiber = _fiber_length_cylinder(V, A)
    if L_fiber is None or L_fiber < L_major:
        # Fall back to cuboid
        L_fiber = _fiber_length_cuboid(V, A)

    if L_fiber is None or L_fiber < L_major:
        return None
    return 1 - L_major / L_fiber

def _curl_3D_skeleton(boundary, height=100):
    # Turn into voxel grid, scale by setting voxel side length
    img = _voxelize(boundary, height=height)

    # Skeletonize & encode skeleton as graph
    skeleton = skeletonize(img).astype(int) ## will use method='lee' for 3D by default
    graph = sknw.build_sknw(skeleton.astype(np.uint16))
    graph = graph.to_undirected()

    # Find longest path (acyclic)
    _, fiber_length = _find_longest_simple_path(graph)

    # Compute major axis using PCA
    major_axis_length = _major_axis(boundary)

    # Return curl (bounded)
    if fiber_length <= major_axis_length or fiber_length == 0:
        return 0.0

    return 1 - (major_axis_length / fiber_length)

def _curl_2D(boundary):
    factor = boundary.length**2 - 16 * boundary.area
    if factor < 0:
        factor = 0
    fibre_length = boundary.area / ((boundary.length - np.sqrt(factor)) / 4)

    _, length = _axes(boundary)
    if fibre_length < length:
        return 0
    else:
        return 1 - length / fibre_length


@d.dedent
def curl(
    adata: AnnData,
    cluster_key: str = "component",
    out_key: str = "curl",
    method_3D: str = "skeleton",
    height: int = 100,
    copy: bool = False,
) -> None | dict[int, float]:
    """
    Compute the curl score of the topological boundaries of sets of cells.

    Given by curl = 1 - major_axis_length / fiber_length.
    In 2D, these are the length of the major axis of the minimum bounding rectangle and the fiber length of the polygon (see Varrone et al., 2023).
    In 3D, two methods are available:
    - 'skeleton': The fiber length is the length of the longest path in the skeleton of the shape, computes as for the linearity score (parameter height required). This method is more likely to return a non-zero value, but is more complex (interpretability).
    - 'shape_comparison': The fiber length is the length of a cylinder (or, as a fallback, cuboid) with the same volume and surface area as the shape. If the shape is straight and elongated, then length_major_axis ≈ fiber_length, so curl ≈ 0. If the shape is coiled, twisted, or compact, then length_major_axis << fiber_length, so curl → 1.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    %(copy)s
    out_key
        Key in :attr:`anndata.AnnData.obs` where the metric values are stored if ``copy = False``.
    method_3D
        Only for 3D shapes: How to calculate the curl score. Either 'skeleton' or 'shape_comparison'.
    height
        Only for 3D shapes: Height of the rasterized image. The lengths of the other side(s) is/are computed automatically to preserve the aspect ratio of the shape. Higher values lead to more precise results but also higher memory usage. Default is 100.

    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the curl score as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['{{out_key}}']`` - - the above mentioned :class:`dict`.

    """
    boundaries = adata.uns[f"shape_{cluster_key}"]["boundary"]
    curl_score = {}
    for cluster, boundary in boundaries.items():
        # Check dimensionality of boundary
        if isinstance(boundary, geometry.Polygon) or isinstance(boundary, geometry.MultiPolygon):
            curl_score[cluster] = _curl_2D(boundary)
        elif isinstance(boundary, trimesh.Trimesh):
            if method_3D == "skeleton":
                curl_score[cluster] = _curl_3D_skeleton(boundary, height=height)
            elif method_3D == "shape_comparison":
                curl_score[cluster] = _curl_3D_shape_comparison(boundary)
            else:
                raise ValueError("Unknown method_3D. Must be either 'skeleton' or 'shape_comparison'.")
        else:
            curl_score[cluster] = np.nan

    if copy:
        return curl_score
    adata.uns[f"shape_{cluster_key}"][out_key] = curl_score


@d.dedent
def purity(
    adata: AnnData,
    cluster_key: str = "component",
    library_key: str = "sample",
    out_key: str = "purity",
    exterior: bool = False,
    copy: bool = False,
) -> None | dict[int, float]:
    """
    Compute the purity of the topological boundaries of sets of cells.

    It computes the purity of each cluster as the ratio between the number of cells of the cluster that are within the boundary and the total number of cells within the boundary.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    library_key
        Key in :attr:`anndata.AnnData.obs` where the sample labels are stored.
    out_key
        Key in :attr:`anndata.AnnData.obs` where the metric values are stored if ``copy = False``.
    exterior
        If ``True``, the computation of the purity ignores the polygon's internal holes.
    %(copy)s
    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the purity as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['{{out_key}}']`` - - the above mentioned :class:`dict`.
    """
    boundaries = adata.uns[f"shape_{cluster_key}"]["boundary"]

    purity_score = {}
    for cluster, boundary in boundaries.items():
        sample = adata[adata.obs[cluster_key] == cluster].obs[library_key].unique()[0]
        adata_sample = adata[adata.obs[library_key] == sample]

        points = adata_sample.obsm["spatial"]
        if points.shape[1] == 2:
            within_mask = np.zeros(points.shape[0], dtype=bool)
            if type(boundary) is geometry.multipolygon.MultiPolygon:
                for p in boundary.geoms:
                    path = Path(np.array(p.exterior.coords.xy).T)
                    within_mask |= np.array(path.contains_points(points))
            else:
                path = Path(np.array(boundary.exterior.coords.xy).T)
                within_mask |= np.array(path.contains_points(points))
                if not exterior:
                    for interior in boundary.interiors:
                        path = Path(np.array(interior.coords.xy).T)
                        within_mask &= ~np.array(path.contains_points(points))
            purity_score[cluster] = np.sum(adata_sample.obs[cluster_key][within_mask] == cluster) / np.sum(within_mask)

        else: # 3D
            is_cluster = adata_sample.obs[cluster_key] == cluster

            # Init masks for all points of other clusters
            not_cluster_within_bounds = np.zeros(points.shape[0], dtype=bool)
            other_cluster_mask = ~is_cluster
            other_cluster_points = points[other_cluster_mask, :]

            # Catch RuntimeWarning: overflow encountered in divide
            # t[nonzero] = (axis_bound[nonzero] - axis_ori[nonzero]) / axis_dir[nonzero]
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", RuntimeWarning)

                    # Try full contains() call
                    result = boundary.contains(other_cluster_points)

                    # If RuntimeWarning caught, do fallback for points near mesh
                    if any(issubclass(warn.category, RuntimeWarning) for warn in w):
                        # Identify "nearby" points for fallback
                        bbox = boundary.bounding_box
                        ((min_x, min_y, min_z), (max_x, max_y, max_z)) = bbox.bounds
                        within_box = (
                            (points[:, 0] >= min_x) & (points[:, 0] <= max_x) &
                            (points[:, 1] >= min_y) & (points[:, 1] <= max_y) &
                            (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
                        )
                        fallback_indices = np.where(within_box & other_cluster_mask)[0]
                        fallback_points = points[fallback_indices]

                        # Per-point containment for only these
                        for i, idx in enumerate(fallback_indices):
                            point = fallback_points[i]
                            try:
                                with warnings.catch_warnings(record=True) as single_w:
                                    warnings.simplefilter("always", RuntimeWarning)
                                    res = boundary.contains([point])[0]
                                    if any(issubclass(w.category, RuntimeWarning) for w in single_w):
                                        res = True  # fallback to inside
                                not_cluster_within_bounds[idx] = res
                            except Exception as e:
                                print(f"Error at point {idx}: {e}")
                                not_cluster_within_bounds[idx] = True

                    else:
                        # If no warning, accept result for full shape
                        not_cluster_within_bounds[other_cluster_mask] = result

            except Exception as e:
                print(f"Purity check failed for shape {cluster}: {e}")
                # Fallback: assume inside
                not_cluster_within_bounds[other_cluster_mask] = True

            purity_score[cluster] = np.sum(is_cluster) / ( np.sum(is_cluster) + np.sum(not_cluster_within_bounds) )

    if copy:
        return purity_score
    adata.uns[f"shape_{cluster_key}"][out_key] = purity_score
