"""Command Line Interface for Geotechnical Reconciliation."""

import argparse
import sys
import json
import os
import pandas as pd
import numpy as np

from core.mesh_handler import load_mesh
from core.section_cutter import (
    generate_sections_along_crest, generate_perpendicular_sections,
    cut_both_surfaces, SectionLine
)
from core.param_extractor import extract_parameters, compare_design_vs_asbuilt
from core.excel_writer import export_results
from core.report_generator import generate_word_report

def main():
    parser = argparse.ArgumentParser(description="Conciliación Geotécnica 3D CLI")
    
    # Inputs
    parser.add_argument("--design", required=True, help="Path to Design Mesh (STL/DXF/OBJ)")
    parser.add_argument("--topo", required=True, help="Path to Topo/Scan Mesh (STL/DXF/OBJ)")
    parser.add_argument("--sections", help="JSON file with section definitions or 'auto'")
    parser.add_argument("--dxf_poly", help="DXF file with polyline for auto sections")
    
    # Params
    parser.add_argument("--spacing", type=float, default=20.0, help="Spacing for auto sections")
    parser.add_argument("--length", type=float, default=150.0, help="Length of sections")
    parser.add_argument("--azimuth", type=float, help="Fixed azimuth for sections (optional)")
    
    # Tolerances
    parser.add_argument("--tol_h", type=float, default=0.5, help="Height tolerance (+/-)")
    parser.add_argument("--tol_a", type=float, default=3.0, help="Angle tolerance (+/- deg)")
    parser.add_argument("--min_berm", type=float, default=8.0, help="Minimum berm width")
    
    # Output
    parser.add_argument("--output", default="results", help="Output directory/prefix")
    parser.add_argument("--report", action="store_true", help="Generate Word report")

    args = parser.parse_args()
    
    # 1. Load Meshes
    print("Loading meshes...")
    try:
        mesh_d = load_mesh(args.design)
        mesh_t = load_mesh(args.topo)
    except Exception as e:
        print(f"Error loading meshes: {e}")
        sys.exit(1)
        
    # 2. Define Sections
    print("Generating sections...")
    sections = []
    
    if args.sections and args.sections.endswith('.json'):
        with open(args.sections, 'r') as f:
            data = json.load(f)
            for s in data:
                sections.append(SectionLine(
                    name=s['name'],
                    origin=np.array(s['origin']),
                    azimuth=s['azimuth'],
                    length=s.get('length', args.length),
                    sector=s.get('sector', "Default")
                ))
    elif args.dxf_poly:
        # Load DXF polyline
        from core.mesh_handler import load_dxf_polyline
        pts = load_dxf_polyline(args.dxf_poly)
        if len(pts) < 2:
            print("Invalid DXF polyline.")
            sys.exit(1)
        
        sections = generate_perpendicular_sections(
            pts, args.spacing, args.length, "DXF_Auto", design_mesh=mesh_d
        )
    else:
        # Auto single line (dummy default)
        print("No section definition provided. Generating demo sections based on mesh bounds center.")
        c = mesh_d.centroid
        p1 = c - np.array([100, 0, 0])
        p2 = c + np.array([100, 0, 0])
        sections = generate_sections_along_crest(
            mesh_d, p1, p2, 5, section_azimuth=0, section_length=args.length
        )

    print(f"Defined {len(sections)} sections.")
    
    # 3. Process
    print("Processing...")
    all_comparisons = []
    params_d_list = []
    params_t_list = []
    all_data = []
    
    tolerances = {
        'bench_height': {'neg': args.tol_h, 'pos': args.tol_h},
        'face_angle': {'neg': args.tol_a, 'pos': args.tol_a},
        'berm_width': {'min': args.min_berm, 'tol': 0.5} # Fixed tol for berm dev
    }
    
    for sec in sections:
        res_d, res_t = cut_both_surfaces(mesh_d, mesh_t, sec)
        
        if res_d and res_t:
            pd = extract_parameters(res_d.distances, res_d.elevations, sec.name, sec.sector)
            pt = extract_parameters(res_t.distances, res_t.elevations, sec.name, sec.sector)
            
            params_d_list.append(pd)
            params_t_list.append(pt)
            
            all_data.append({
                'section_name': sec.name,
                'params_design': pd,
                'params_topo': pt,
                'profile_d': (res_d.distances, res_d.elevations),
                'profile_t': (res_t.distances, res_t.elevations)
            })
            
            comps = compare_design_vs_asbuilt(pd, pt, tolerances)
            all_comparisons.extend(comps)
            
    # 4. Export
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
        
    excel_path = os.path.join(args.output, "conciliacion_results.xlsx")
    print(f"Exporting Excel to {excel_path}...")
    export_results(all_comparisons, params_d_list, params_t_list,
                   tolerances, excel_path, 
                   {'project': 'CLI Run', 'author': 'Automated'})
                   
    if args.report:
        word_path = os.path.join(args.output, "conciliacion_report.docx")
        print(f"Exporting Word Report to {word_path}...")
        generate_word_report(all_comparisons, all_data, word_path,
                             {'project': 'CLI Run', 'author': 'Automated'})

    print("Done.")

if __name__ == "__main__":
    main()
