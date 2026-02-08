"""Mesh loading, decimation, bounds extraction and conversion to plotly."""

import trimesh
import numpy as np
import plotly.graph_objects as go


def load_mesh(filepath):
    """Load a 3D surface mesh from STL, OBJ, or PLY file."""
    mesh = trimesh.load(filepath)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
        raise ValueError("El archivo no contiene una malla 3D válida.")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("La malla está vacía (0 vértices o 0 caras).")
    return mesh


def get_mesh_bounds(mesh):
    """Get bounding box and statistics for a mesh."""
    bounds = mesh.bounds  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    center = mesh.centroid
    return {
        'xmin': float(bounds[0][0]), 'xmax': float(bounds[1][0]),
        'ymin': float(bounds[0][1]), 'ymax': float(bounds[1][1]),
        'zmin': float(bounds[0][2]), 'zmax': float(bounds[1][2]),
        'center': center,
        'n_faces': len(mesh.faces),
        'n_vertices': len(mesh.vertices),
    }


def _vertex_clustering(mesh, target_faces):
    """
    Decimate via vertex clustering: group nearby vertices into grid cells,
    merge them, and rebuild faces. Preserves mesh connectivity.
    """
    target_verts = max(target_faces // 2, 500)
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]

    # Cell size so that grid has ~target_verts cells
    volume = float(np.prod(extents))
    cell_size = (volume / target_verts) ** (1.0 / 3.0) if volume > 0 else 1.0

    # Quantize vertices to grid cells
    grid = ((mesh.vertices - bounds[0]) / cell_size).astype(np.int32)

    # Encode (gx, gy, gz) into a single key per vertex
    mx = int(grid[:, 0].max()) + 1
    my = int(grid[:, 1].max()) + 1
    keys = grid[:, 0] * my * (int(grid[:, 2].max()) + 1) + \
           grid[:, 1] * (int(grid[:, 2].max()) + 1) + grid[:, 2]

    # Map each original vertex to a new cluster index
    unique_keys, inverse = np.unique(keys, return_inverse=True)

    # Average vertex positions per cluster
    new_verts = np.zeros((len(unique_keys), 3))
    counts = np.zeros(len(unique_keys))
    np.add.at(new_verts, inverse, mesh.vertices)
    np.add.at(counts, inverse, 1)
    new_verts /= counts[:, None]

    # Remap face indices
    new_faces = inverse[mesh.faces]

    # Remove degenerate faces (two or more vertices collapsed to same cluster)
    valid = ((new_faces[:, 0] != new_faces[:, 1]) &
             (new_faces[:, 1] != new_faces[:, 2]) &
             (new_faces[:, 0] != new_faces[:, 2]))
    new_faces = new_faces[valid]

    # Remove duplicate faces
    sorted_f = np.sort(new_faces, axis=1)
    _, unique_idx = np.unique(sorted_f, axis=0, return_index=True)
    new_faces = new_faces[unique_idx]

    return trimesh.Trimesh(vertices=new_verts, faces=new_faces)


def decimate_mesh(mesh, target_faces):
    """Reduce mesh face count for visualization performance."""
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(target_faces)
    except (ImportError, Exception):
        return _vertex_clustering(mesh, target_faces)


def mesh_to_plotly(mesh, name, color, opacity):
    """Convert a trimesh mesh to a plotly Mesh3d trace."""
    vertices = mesh.vertices
    faces = mesh.faces
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        name=name,
        color=color,
        opacity=opacity,
        showlegend=True,
    )
