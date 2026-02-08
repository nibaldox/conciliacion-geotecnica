"""Section definition, generation and mesh cutting."""

import numpy as np
from dataclasses import dataclass
import trimesh


@dataclass
class SectionLine:
    """A vertical section defined by origin, azimuth and length."""
    name: str
    origin: np.ndarray
    azimuth: float
    length: float
    sector: str = ""


@dataclass
class ProfileResult:
    """Result of cutting a mesh with a section: 2D profile."""
    distances: np.ndarray
    elevations: np.ndarray


def azimuth_to_direction(azimuth_deg):
    """Convert azimuth (degrees from North, clockwise) to 2D direction vector."""
    az_rad = np.radians(azimuth_deg)
    return np.array([np.sin(az_rad), np.cos(az_rad)])


def cut_mesh_with_section(mesh, section):
    """
    Cut a mesh with a vertical plane defined by a SectionLine.
    Returns a ProfileResult with distances and elevations, or None.
    """
    direction = azimuth_to_direction(section.azimuth)

    # Plane normal (perpendicular to direction in XY plane)
    plane_normal = np.array([direction[1], -direction[0], 0.0])
    plane_origin = np.array([section.origin[0], section.origin[1], 0.0])

    try:
        lines = trimesh.intersections.mesh_plane(
            mesh, plane_normal, plane_origin
        )
    except Exception:
        return None

    if lines is None or len(lines) == 0:
        return None

    # Collect all intersection points
    points = []
    for segment in lines:
        for point in segment:
            points.append(point)

    points = np.array(points)

    # Project onto section direction to get distance along section
    origin_2d = section.origin
    dists = ((points[:, 0] - origin_2d[0]) * direction[0] +
             (points[:, 1] - origin_2d[1]) * direction[1])
    elevs = points[:, 2]

    # Filter by section length
    half_len = section.length / 2
    mask = (dists >= -half_len) & (dists <= half_len)
    dists = dists[mask]
    elevs = elevs[mask]

    if len(dists) < 2:
        return None

    # Sort by distance
    order = np.argsort(dists)
    dists = dists[order]
    elevs = elevs[order]

    # Remove near-duplicate distances by rounding and averaging elevations
    rounded = np.round(dists, 3)
    unique_dists, inv = np.unique(rounded, return_inverse=True)
    unique_elevs = np.zeros(len(unique_dists))
    counts = np.zeros(len(unique_dists))
    for idx, uid in enumerate(inv):
        unique_elevs[uid] += elevs[idx]
        counts[uid] += 1
    unique_elevs /= counts

    return ProfileResult(distances=unique_dists, elevations=unique_elevs)


def cut_both_surfaces(mesh_design, mesh_topo, section):
    """Cut both design and topo meshes with the same section."""
    pd = cut_mesh_with_section(mesh_design, section)
    pt = cut_mesh_with_section(mesh_topo, section)
    return pd, pt


def compute_local_azimuth(mesh, point_xy, radius=50.0):
    """
    Compute the steepest descent azimuth at a point on the mesh surface.
    Fits a plane to nearby vertices and returns the downhill direction.
    """
    verts = mesh.vertices
    dx = verts[:, 0] - point_xy[0]
    dy = verts[:, 1] - point_xy[1]
    dists_sq = dx ** 2 + dy ** 2

    mask = dists_sq < radius ** 2
    if mask.sum() < 10:
        mask = dists_sq < (radius * 3) ** 2
        if mask.sum() < 10:
            return 0.0

    local_verts = verts[mask]

    # Fit plane z = a*x + b*y + c via least squares
    A = np.column_stack([local_verts[:, 0], local_verts[:, 1],
                         np.ones(len(local_verts))])
    z = local_verts[:, 2]
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    # Gradient (steepest ascent) = (a, b); descent = (-a, -b)
    grad_x, grad_y = coeffs[0], coeffs[1]

    # Azimuth from North, clockwise: arctan2(east_component, north_component)
    azimuth = np.degrees(np.arctan2(-grad_x, -grad_y)) % 360
    return float(azimuth)


def generate_sections_along_crest(mesh, start_point, end_point, n_sections,
                                  section_azimuth, section_length,
                                  sector_name=""):
    """Generate evenly spaced sections along a line (e.g., pit crest)."""
    sections = []
    for i in range(n_sections):
        t = i / (n_sections - 1) if n_sections > 1 else 0.5
        origin = start_point + t * (end_point - start_point)
        sections.append(SectionLine(
            name=f"S-{i+1:02d}",
            origin=origin,
            azimuth=section_azimuth,
            length=section_length,
            sector=sector_name,
        ))
    return sections


def generate_perpendicular_sections(points, spacing, section_length,
                                    sector_name="", auto_azimuth_mesh=None):
    """
    Generate sections perpendicular to a polyline at specified spacing.

    Parameters:
        points: Nx2 array of (X, Y) coordinates defining the evaluation line
        spacing: Distance between sections in meters
        section_length: Length of each section in meters
        sector_name: Sector name for labeling
        auto_azimuth_mesh: If provided, compute azimuth from mesh slope
                           instead of line perpendicular
    Returns:
        List of SectionLine objects
    """
    points = np.atleast_2d(points)
    if len(points) < 2:
        return []

    # Cumulative distance along polyline
    diffs = np.diff(points, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_dist = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_dist[-1]

    if total_length < 1e-6:
        return []

    # Section positions along the polyline
    if total_length < spacing:
        section_dists = [total_length / 2]
    else:
        section_dists = np.arange(spacing / 2, total_length, spacing)

    sections = []
    for i, d in enumerate(section_dists):
        # Find which segment we're on
        seg_idx = int(np.searchsorted(cum_dist, d, side='right')) - 1
        seg_idx = max(0, min(seg_idx, len(points) - 2))

        # Interpolate position
        t = ((d - cum_dist[seg_idx]) / seg_lengths[seg_idx]
             if seg_lengths[seg_idx] > 0 else 0)
        origin = points[seg_idx] + t * diffs[seg_idx]

        if auto_azimuth_mesh is not None:
            az = compute_local_azimuth(auto_azimuth_mesh, origin)
        else:
            # Perpendicular to the polyline tangent
            tangent = diffs[seg_idx]
            tangent_az = np.degrees(np.arctan2(tangent[0], tangent[1])) % 360
            az = (tangent_az + 90) % 360

        sections.append(SectionLine(
            name=f"S-{i + 1:02d}",
            origin=origin,
            azimuth=az,
            length=section_length,
            sector=sector_name,
        ))

    return sections
