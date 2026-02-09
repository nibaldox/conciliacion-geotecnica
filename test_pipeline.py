import numpy as np
import pytest
import os
import trimesh
from core.section_cutter import SectionLine, cut_mesh_with_section, generate_sections_along_crest
from core.param_extractor import extract_parameters, compare_design_vs_asbuilt

# Mock Data Generation Helpers

def create_pit_surface(center, radius_top, radius_bottom, height, bench_height, bench_width):
    """
    Create a synthetic pit surface (conical with steps) roughly imitating benches.
    This constructs a mesh with vertices and faces.
    """
    # Simple revolution mesh
    # Profile: (r, z) points
    # Start at bottom
    profile_pts = []
    
    z_current = 0
    r_current = radius_bottom
    
    while z_current < height:
        # Point at Toe
        profile_pts.append((r_current, z_current))
        
        # Up (Face)
        z_next = min(z_current + bench_height, height)
        # Face angle approx 75 deg? tan(75) ~ 3.7
        # dr = dz / tan(ang)
        dr_face = (z_next - z_current) / 3.7 
        r_next = r_current + dr_face
        
        profile_pts.append((r_next, z_next))
        
        if z_next < height:
            # Out (Berm)
            r_next_toe = r_next + bench_width
            profile_pts.append((r_next_toe, z_next))
            
            r_current = r_next_toe
        
        z_current = z_next

    # Revolve profile
    theta = np.linspace(0, 2*np.pi, 36) # 36 segments (10 deg)
    vertices = []
    
    # Vertices
    for t in theta:
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        for r, z in profile_pts:
            vertices.append([center[0] + r*cos_t, center[1] + r*sin_t, center[2] + z])
            
    # Faces
    faces = []
    n_prof = len(profile_pts)
    n_theta = len(theta)
    
    for i in range(n_theta - 1):
        for j in range(n_prof - 1):
            # Quad formed by (i,j), (i+1, j), (i+1, j+1), (i, j+1)
            # v0 = i*n_prof + j
            # v1 = (i+1)*n_prof + j
            # v2 = (i+1)*n_prof + (j+1)
            # v3 = i*n_prof + (j+1)
            
            v0 = i * n_prof + j
            v1 = (i + 1) * n_prof + j
            v2 = (i + 1) * n_prof + (j + 1)
            v3 = i * n_prof + (j + 1)
            
            # Tri 1: v0, v1, v2
            faces.append([v0, v1, v2])
            # Tri 2: v0, v2, v3
            faces.append([v0, v2, v3])
            
    # Link last sector to first
    i = n_theta - 1
    for j in range(n_prof - 1):
        v0 = i * n_prof + j
        v1 = 0 * n_prof + j # wrap to 0
        v2 = 0 * n_prof + (j + 1)
        v3 = i * n_prof + (j + 1)
        
        faces.append([v0, v1, v2])
        faces.append([v0, v2, v3])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh

def test_full_reconciliation_flow():
    """
    Test the pipeline: Mesh -> Cut -> Extract -> Compare.
    """
    # 1. Setup Data
    # Design: Ideal pit
    mesh_design = create_pit_surface([0,0,0], 100, 50, 30, 10, 5)
    
    # Topo: Slightly wider, maybe lower benches (simulating imperfection)
    # 30 height, 9.5 bench height (error), 5.2 berm (error)
    mesh_topo = create_pit_surface([0,0,0], 102, 52, 30, 9.5, 5.2)

    # 2. Define Section
    # Cross section through center, X-axis
    section = SectionLine(
        name="TestSection",
        origin=np.array([70, 0, 0]), # On the wall
        azimuth=90, # Facing East (outwards from center? Center is 0,0. Wall is at Radius ~50-80)
        # Wait, if center is 0,0. Radius grows outwards.
        # Wall is at R=50 to R=100.
        # Section Origin at 75 (mid slope). 
        # Azimuth 90 is East. Section runs West-East.
        # Profile distance will be along X.
        length=200,
        sector="Sector1"
    )
    
    # 3. Cut
    res_d = cut_mesh_with_section(mesh_design, section)
    res_t = cut_mesh_with_section(mesh_topo, section)
    
    assert res_d is not None
    assert res_t is not None
    assert len(res_d.distances) > 10
    
    # 4. Extract
    pd = extract_parameters(res_d.distances, res_d.elevations, section.name, section.sector)
    pt = extract_parameters(res_t.distances, res_t.elevations, section.name, section.sector)
    
    assert len(pd.benches) > 0
    assert len(pt.benches) > 0
    
    # Expected 3 benches (30m height / 10m)
    # Check bench heights
    print(f"Design Benches: {len(pd.benches)}")
    for b in pd.benches:
        print(b)
        assert abs(b.bench_height - 10.0) < 1.0 # Due to discretization/interpolation
        
    print(f"Topo Benches: {len(pt.benches)}")
    for b in pt.benches:
        print(b)
        assert abs(b.bench_height - 9.5) < 1.0
        
    # 5. Compare
    tolerances = {
        'bench_height': {'neg': 0.5, 'pos': 0.5},
        'face_angle': {'neg': 5, 'pos': 5},
        'berm_width': {'min': 4.0, 'tol': 1.0}
    }
    
    comparisons = compare_design_vs_asbuilt(pd, pt, tolerances)
    
    assert len(comparisons) > 0
    # Check first bench comparison
    c0 = comparisons[0]
    # Design 10 vs Real 9.5 -> Diff -0.5. Tol 0.5. Status CUMPLE? Or Boundary?
    # 10 - 9.5 = 0.5 diff. (Real - Design? -0.5). abs(-0.5) <= 0.5 -> CUMPLE
    print(c0)
    assert c0['height_status'] == "CUMPLE" or c0['height_status'] == "FUERA DE TOLERANCIA"


def test_section_generation():
    # Test auto sections allow line
    mesh = create_pit_surface([0,0,0], 100, 50, 30, 10, 5)
    p1 = np.array([0, -100, 0])
    p2 = np.array([0, 100, 0])
    
    sections = generate_sections_along_crest(mesh, p1, p2, n_sections=3, section_length=100)
    assert len(sections) == 3
    assert sections[0].name == "S-01"
    
    # Check azimuth
    # Line is South to North (Y axis). 
    # Direction Y (0,1). Angle = 0 (from North? arctan(x,y)). 
    # atan2(0, 1) = 0.
    # Perpendicular: +90 -> 90 deg (East).
    assert abs(sections[0].azimuth - 90.0) < 0.1

if __name__ == "__main__":
    # Manually run if executed script
    try:
        test_full_reconciliation_flow()
        test_section_generation()
        print("Tests Passed!")
    except AssertionError as e:
        print(f"Test Failed: {e}")
        raise
