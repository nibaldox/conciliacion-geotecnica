import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import trimesh
import os
import tempfile
import json
from io import BytesIO

from core.mesh_handler import load_mesh, get_mesh_bounds, mesh_to_plotly, decimate_mesh, load_dxf_polyline
from core.section_cutter import (
    compute_local_azimuth, generate_sections_along_crest, 
    generate_perpendicular_sections, cut_both_surfaces, SectionLine
)
from core.param_extractor import extract_parameters, compare_design_vs_asbuilt
from core.excel_writer import export_results
from core.report_generator import generate_word_report, generate_section_images_zip

# --- CONFIG ---
st.set_page_config(page_title="Conciliación Geotécnica 3D", layout="wide")

if 'sections' not in st.session_state:
    st.session_state.sections = []
if 'mesh_design' not in st.session_state:
    st.session_state.mesh_design = None
if 'mesh_topo' not in st.session_state:
    st.session_state.mesh_topo = None
if 'comparisons' not in st.session_state:
    st.session_state.comparisons = []
if 'params_design' not in st.session_state:
    st.session_state.params_design = []
if 'params_topo' not in st.session_state:
    st.session_state.params_topo = []
if 'all_data' not in st.session_state:
    st.session_state.all_data = [] # Store all data for reporting

# --- SIDEBAR ---
with st.sidebar:
    st.title("Configuración")
    
    st.subheader("1. Cargar Archivos")
    file_design = st.file_uploader("Superficie Diseño (STL/DXF/PLY/OBJ)", type=["stl", "dxf", "ply", "obj"])
    file_topo = st.file_uploader("Superficie Topo/Scan (STL/DXF/PLY/OBJ)", type=["stl", "dxf", "ply", "obj"])
    
    st.subheader("2. Parámetros de Análisis")
    tol_height = st.slider("Tol. Altura Banco (m)", 0.1, 2.0, 0.5)
    tol_angle = st.slider("Tol. Ángulo Cara (°)", 1.0, 10.0, 3.0)
    tol_berm = st.slider("Tol. Ancho Berma (m)", 0.1, 3.0, 0.5)
    
    min_berm_width = st.number_input("Ancho Mínimo Berma (m)", value=8.0)

    st.subheader("3. Definición de Secciones")
    section_mode = st.radio("Modo de Sección", 
                            ["Manual (Click en Mapa)", "Automático (Línea)", "Automático (DXF)"])
    
    section_len = st.number_input("Longitud Sección (m)", value=150.0)
    
    if section_mode == "Automático (Línea)":
        st.info("Define línea de cresta mediante coordenadas o clicks (futuro).")
        n_sections = st.number_input("N° Secciones", value=5, min_value=1)
        # Placeholder for start/end points
        c1 = st.text_input("Inicio (x,y)", "0,0")
        c2 = st.text_input("Fin (x,y)", "100,100")
        
        # New: Option for Azimuth
        use_auto_azimuth = st.checkbox("Calcular Azimut Automático (Perpendicular)", value=True)
        manual_azimuth = 0.0
        if not use_auto_azimuth:
            manual_azimuth = st.number_input("Azimut Manual (°)", value=90.0, min_value=0.0, max_value=360.0)

    elif section_mode == "Automático (DXF)":
        dxf_file = st.file_uploader("Cargar Polilínea DXF (Ejes)", type=["dxf"])
        spacing = st.number_input("Espaciamiento (m)", value=20.0, min_value=1.0)
    
    # Action Buttons
    if st.button("Ejecutar Análisis"):
        if st.session_state.mesh_design and st.session_state.mesh_topo and st.session_state.sections:
            with st.spinner("Procesando secciones..."):
                # Reset results
                st.session_state.comparisons = []
                st.session_state.params_design = []
                st.session_state.params_topo = []
                st.session_state.all_data = []
                
                tolerances = {
                    'bench_height': {'neg': tol_height, 'pos': tol_height},
                    'face_angle': {'neg': tol_angle, 'pos': tol_angle},
                    'berm_width': {'min': min_berm_width, 'tol': tol_berm}
                }
                
                for sec in st.session_state.sections:
                    # Cut
                    res_d, res_t = cut_both_surfaces(
                        st.session_state.mesh_design, 
                        st.session_state.mesh_topo, 
                        sec
                    )
                    
                    if res_d and res_t:
                        # Extract params
                        pd = extract_parameters(res_d.distances, res_d.elevations, 
                                                sec.name, sec.sector)
                        pt = extract_parameters(res_t.distances, res_t.elevations, 
                                                sec.name, sec.sector)
                        
                        st.session_state.params_design.append(pd)
                        st.session_state.params_topo.append(pt)
                        
                        # Store detailed data for reporting/plotting
                        st.session_state.all_data.append({
                            'section_name': sec.name,
                            'params_design': pd,
                            'params_topo': pt,
                            'profile_d': (res_d.distances, res_d.elevations),
                            'profile_t': (res_t.distances, res_t.elevations)
                        })
                        
                        # Compare - NOW USING pd and pt directly
                        comps = compare_design_vs_asbuilt(pd, pt, tolerances)
                        st.session_state.comparisons.extend(comps)
                
                st.success("Análisis completado!")
        else:
            st.error("Faltan datos (mallas o secciones).")

    if st.button("Limpiar Secciones"):
        st.session_state.sections = []


# --- MAIN AREA ---

def load_uploaded_mesh(uploaded_file):
    suffix = "." + uploaded_file.name.split('.')[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name
    try:
        mesh = load_mesh(path)
        os.unlink(path)
        return mesh
    except Exception as e:
        st.error(f"Error cargando malla: {e}")
        return None

# Load Meshes
if file_design and st.session_state.mesh_design is None:
    st.session_state.mesh_design = load_uploaded_mesh(file_design)
    
if file_topo and st.session_state.mesh_topo is None:
    st.session_state.mesh_topo = load_uploaded_mesh(file_topo)

# Visualization
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Visualización 3D / Planta")
    
    if st.session_state.mesh_design:
        # Create Plotly Scene
        fig = go.Figure()
        
        # Design Mesh (Decimated for speed)
        d_dec = decimate_mesh(st.session_state.mesh_design, 5000)
        fig.add_trace(mesh_to_plotly(d_dec, "Diseño", "cyan", 0.5))
        
        # Topo Mesh
        if st.session_state.mesh_topo:
            t_dec = decimate_mesh(st.session_state.mesh_topo, 5000)
            fig.add_trace(mesh_to_plotly(t_dec, "Topo", "red", 0.5))
            
        # Draw Sections
        for sec in st.session_state.sections:
            # Simple line for section
            # Origin +/- length * direction
            # We need standard trigonometry to draw the line in 3D (z=center z)
            # Just draw on Z=center to be visible
            
            # Draw a vertical line or local axis? 
            # Let's draw the cut line on the XY plane projected to Z max
            
            dz = get_mesh_bounds(d_dec)['zmax'] + 10
            
            rad = np.radians(sec.azimuth)
            dx = np.sin(rad) * (sec.length/2)
            dy = np.cos(rad) * (sec.length/2)
            
            p1 = sec.origin - np.array([dx, dy, 0])
            p2 = sec.origin + np.array([dx, dy, 0])
            
            fig.add_trace(go.Scatter3d(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[dz, dz],
                mode='lines', name=sec.name, line=dict(color='yellow', width=5)
            ))
            
        
        # Camera options
        fig.update_layout(scene_aspectmode='data', height=600)
        
        # Event handling for clicks? 
        # Streamlit-plotly events are limited. 
        # We can simulate manual selection by inputting coordinates in Sidebar 
        # or implementing a sophisticated callbacks approach (complex).
        # For MVP: "Manual" mode uses coordinates from interaction? 
        # Actually standard st.plotly_chart doesn't return click coords reliably without component.
        # We will assume User enters coordinates manually for "Single Section" or automated.
        
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Gestión de Secciones")
    
    # Manual Section Entry
    if section_mode == "Manual (Click en Mapa)":
        st.write("Ingrese coordenadas centro:")
        mx = st.number_input("X", value=0.0)
        my = st.number_input("Y", value=0.0)
        maz = st.number_input("Azimut (°)", value=0.0)
        
        if st.button("Agregar Sección"):
            s_name = f"S-{len(st.session_state.sections)+1}"
            st.session_state.sections.append(SectionLine(
                name=s_name, origin=np.array([mx, my, 0]), 
                azimuth=maz, length=section_len, sector="Manual"
            ))
            st.success(f"Sección {s_name} agregada.")
            
    elif section_mode == "Automático (Línea)":
        if st.button("Generar Secciones (Línea)"):
            try:
                p_start = np.fromstring(c1, sep=',')
                p_end = np.fromstring(c2, sep=',')
                
                # Logic for Azimuth
                calc_azimuth = None
                if not use_auto_azimuth:
                    calc_azimuth = manual_azimuth
                    
                new_secs = generate_sections_along_crest(
                    st.session_state.mesh_design, p_start, p_end, 
                    int(n_sections), section_azimuth=calc_azimuth, section_length=section_len,
                    sector_name="AutoLine"
                )
                st.session_state.sections.extend(new_secs)
                st.success(f"Generadas {len(new_secs)} secciones.")
            except Exception as e:
                st.error(f"Error generando secciones: {e}")

    elif section_mode == "Automático (DXF)" and 'dxf_file' in locals() and dxf_file:
         if st.button("Generar desde DXF"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                tmp.write(dxf_file.getvalue())
                dxf_path = tmp.name
            
            try:
                # Load polyline points
                pts = load_dxf_polyline(dxf_path)
                if len(pts) > 0:
                    # Generate sections
                    # Note: We pass the design mesh to calculate local slope if possible
                    # But load_dxf_polyline returns 2D points. section_cutter expects 2D.
                    # azimuth logic needs mesh if we want "downhill" direction.
                    
                    new_secs = generate_perpendicular_sections(
                        pts, spacing, section_len, sector_name="DXF",
                        design_mesh=st.session_state.mesh_design
                    )
                    st.session_state.sections.extend(new_secs)
                    st.success(f"Generadas {len(new_secs)} secciones desde DXF.")
                else:
                    st.error("No se encontró polilínea válida en DXF.")
            except Exception as e:
                st.error(f"Error procesando DXF: {e}")
            finally:
                if os.path.exists(dxf_path):
                    os.unlink(dxf_path)

    # List Sections
    if st.session_state.sections:
        st.write(f"Total: {len(st.session_state.sections)}")
        df_sec = pd.DataFrame([vars(s) for s in st.session_state.sections])
        # Drop origin numpy array for display
        df_sec['origin'] = df_sec['origin'].apply(lambda x: str(x[:2]))
        st.dataframe(df_sec[['name', 'origin', 'azimuth', 'sector']], height=200)


# --- RESULTS AREA ---
if st.session_state.comparisons:
    st.divider()
    st.header("Resultados de Conciliación")
    
    # 1. Dashboard Metrics
    comps = st.session_state.comparisons
    n_total = len(comps) * 3 # Height, Angle, Berm
    n_ok = sum(1 for c in comps for k in ['height_status','angle_status','berm_status'] if c[k] == "CUMPLE")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Bancos Evaluados", len(comps))
    m2.metric("Cumplimiento Global", f"{(n_ok/n_total*100):.1f}%")
    
    # 2. Detailed Table
    df_res = pd.DataFrame(comps)
    st.dataframe(df_res.style.applymap(lambda v: "background-color: #ffcdd2" if "NO" in str(v) 
                                       else ("background-color: #fff9c4" if "FUERA" in str(v) 
                                             else "background-color: #c8e6c9") if isinstance(v, str) else "",
                                       subset=['height_status', 'angle_status', 'berm_status']))
                                       
    # 3. Profile Plots Explorer
    st.subheader("Explorador de Perfiles")
    
    # Select section to view
    sec_names = [d['section_name'] for d in st.session_state.all_data]
    if sec_names:
        sel_sec = st.selectbox("Seleccionar Sección", sec_names)
        
        # Get data
        data_idx = sec_names.index(sel_sec)
        data_item = st.session_state.all_data[data_idx]
        
        pd_obj = data_item['params_design']
        pt_obj = data_item['params_topo']
        prof_d = data_item['profile_d']
        prof_t = data_item['profile_t']
        
        # Plot with Matplotlib (reusing report logic or new simple plot)
        # Let's use Plotly for interactivity here
        fig_prof = go.Figure()
        
        # Design
        fig_prof.add_trace(go.Scatter(x=prof_d[0], y=prof_d[1], name="Diseño", line=dict(color='cyan')))
        # Topo
        fig_prof.add_trace(go.Scatter(x=prof_t[0], y=prof_t[1], name="Real", line=dict(color='red')))
        
        # Reconciled (Points)
        if pt_obj.benches:
            rx, ry = [], []
            for b in pt_obj.benches:
                # Crest
                fig_prof.add_trace(go.Scatter(x=[b.crest_distance], y=[b.crest_elevation], 
                                              mode='markers+text', text=[f"C{b.bench_number}"],
                                              marker=dict(color='green', size=10, symbol='triangle-down'),
                                              name='Cresta Real', showlegend=False))
                # Toe
                fig_prof.add_trace(go.Scatter(x=[b.toe_distance], y=[b.toe_elevation],
                                              mode='markers', 
                                              marker=dict(color='blue', size=8, symbol='triangle-up'),
                                              name='Pata Real', showlegend=False))
        
        fig_prof.update_layout(title=f"Perfil {sel_sec}", xaxis_title="Distancia", yaxis_title="Elevación",
                               height=500)
        st.plotly_chart(fig_prof, use_container_width=True)

    # 4. Exports
    st.subheader("Exportar Reportes")
    
    # Excel
    if st.button("Generar Reporte Excel"):
        out_buffer = BytesIO()
        # Create temp file to write then read to bytes
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tpath = tmp.name
        
        info = {
            'project': "Demo Conciliacion", 
            'author': "User", 
            'date': "2024-05-21"
        }
        tolerances_export = {
            'bench_height': {'neg': tol_height, 'pos': tol_height},
            'face_angle': {'neg': tol_angle, 'pos': tol_angle},
            'berm_width': {'min': min_berm_width, 'tol': tol_berm},
             # Missing others but safe default
        }
        
        export_results(st.session_state.comparisons, 
                       st.session_state.params_design, 
                       st.session_state.params_topo, 
                       tolerances_export, tpath, info)
                       
        with open(tpath, "rb") as f:
            st.download_button("Descargar Excel", f, file_name="conciliacion.xlsx")
            
    # Word
    if st.button("Generar Reporte Word"):
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            wpath = tmp.name
            
        info = {'project': "Demo", 'author': "User"}
        generate_word_report(st.session_state.comparisons, st.session_state.all_data, wpath, info)
        
        with open(wpath, "rb") as f:
            st.download_button("Descargar Word", f, file_name="reporte.docx")

    # Images ZIP
    if st.button("Exportar Imágenes de Secciones (ZIP)"):
        with st.spinner("Generando imágenes..."):
            zip_buf = generate_section_images_zip(st.session_state.all_data)
            st.download_button(
                label="Descargar Imágenes (ZIP)",
                data=zip_buf,
                file_name="secciones_conciliacion.zip",
                mime="application/zip"
            )
