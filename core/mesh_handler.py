"""Mesh loading, decimation, bounds extraction and conversion to plotly."""

import trimesh
import numpy as np
import plotly.graph_objects as go


def load_mesh(filepath):
    """Load a 3D surface mesh from STL, OBJ, or PLY file."""
    mesh = trimesh.load(filepath)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
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


def decimate_mesh(mesh, target_faces):
    """Reduce mesh face count for visualization performance."""
    if len(mesh.faces) <= target_faces:
        return mesh
    return mesh.simplify_quadric_decimation(target_faces)


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
