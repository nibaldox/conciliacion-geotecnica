"""
Aplicación Streamlit para Conciliación Geotécnica: Diseño vs As-Built
Carga superficies STL, genera secciones, extrae parámetros y exporta a Excel.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile
import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from core import (
    load_mesh, get_mesh_bounds, mesh_to_plotly, decimate_mesh,
    SectionLine, cut_mesh_with_section, cut_both_surfaces,
    extract_parameters, compare_design_vs_asbuilt, export_results,
)

st.set_page_config(page_title="Conciliación Geotécnica", page_icon="⛏️", layout="wide")

# =====================================================
# CSS
# =====================================================
st.markdown("""
<style>
.main-title { font-size: 2rem; font-weight: bold; color: #2F5496; text-align: center; margin-bottom: 0.5rem; }
.subtitle { font-size: 1.1rem; color: #666; text-align: center; margin-bottom: 1.5rem; }
.metric-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 1rem; border-radius: 10px; text-align: center; margin: 0.5rem 0;
    border-left: 4px solid #2F5496;
}
.status-ok { background-color: #C6EFCE; color: #006100; padding: 4px 12px; border-radius: 4px; font-weight: bold; }
.status-warn { background-color: #FFEB9C; color: #9C5700; padding: 4px 12px; border-radius: 4px; font-weight: bold; }
.status-nok { background-color: #FFC7CE; color: #9C0006; padding: 4px 12px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⛏️ Conciliación Geotécnica: Diseño vs As-Built</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Extracción automática de parámetros desde superficies 3D (STL)</div>', unsafe_allow_html=True)

# =====================================================
# SESSION STATE
# =====================================================
defaults = {
    'mesh_design': None, 'mesh_topo': None,
    'bounds_design': None, 'bounds_topo': None,
    'sections': [], 'profiles_design': [], 'profiles_topo': [],
    'params_design': [], 'params_topo': [],
    'comparison_results': [], 'step': 1,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =====================================================
# SIDEBAR: CONFIGURACIÓN
# =====================================================
with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.subheader("📐 Tolerancias")
    tol_h_neg = st.number_input("Altura banco: Tol. (-) m", value=1.0, step=0.5, key="tol_h_neg")
    tol_h_pos = st.number_input("Altura banco: Tol. (+) m", value=1.5, step=0.5, key="tol_h_pos")
    tol_a_neg = st.number_input("Ángulo cara: Tol. (-) °", value=5.0, step=1.0, key="tol_a_neg")
    tol_a_pos = st.number_input("Ángulo cara: Tol. (+) °", value=5.0, step=1.0, key="tol_a_pos")
    tol_b_neg = st.number_input("Berma: Tol. (-) m", value=1.0, step=0.5, key="tol_b_neg")
    tol_b_pos = st.number_input("Berma: Tol. (+) m", value=2.0, step=0.5, key="tol_b_pos")
    tol_ir_neg = st.number_input("Áng. Inter-Rampa: Tol. (-) °", value=3.0, step=1.0, key="tol_ir_neg")
    tol_ir_pos = st.number_input("Áng. Inter-Rampa: Tol. (+) °", value=2.0, step=1.0, key="tol_ir_pos")
    
    st.subheader("🔧 Detección de Bancos")
    face_threshold = st.slider("Ángulo mínimo cara (°)", 30, 60, 40)
    berm_threshold = st.slider("Ángulo máximo berma (°)", 5, 30, 20)
    resolution = st.slider("Resolución de perfil (m)", 0.1, 2.0, 0.5)
    
    st.subheader("📋 Información del Proyecto")
    project_name = st.text_input("Proyecto", "")
    operation = st.text_input("Operación", "")
    phase = st.text_input("Fase / Pit", "")
    author = st.text_input("Elaborado por", "")

tolerances = {
    'bench_height': {'neg': tol_h_neg, 'pos': tol_h_pos},
    'face_angle': {'neg': tol_a_neg, 'pos': tol_a_pos},
    'berm_width': {'neg': tol_b_neg, 'pos': tol_b_pos},
    'inter_ramp_angle': {'neg': tol_ir_neg, 'pos': tol_ir_pos},
    'overall_angle': {'neg': 2.0, 'pos': 2.0},
}

# =====================================================
# PASO 1: CARGA DE SUPERFICIES
# =====================================================
st.header("📁 Paso 1: Cargar Superficies STL")

col1, col2 = st.columns(2)
with col1:
    st.subheader("🔵 Superficie de Diseño")
    file_design = st.file_uploader("Cargar STL de Diseño", type=["stl", "obj", "ply"], key="design_file")

with col2:
    st.subheader("🟢 Superficie Topográfica (As-Built)")
    file_topo = st.file_uploader("Cargar STL Topografía Real", type=["stl", "obj", "ply"], key="topo_file")

if file_design and file_topo:
    try:
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            f.write(file_design.read()); f_design = f.name
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            f.write(file_topo.read()); f_topo = f.name
        
        with st.spinner("Cargando superficies..."):
            st.session_state.mesh_design = load_mesh(f_design)
            st.session_state.mesh_topo = load_mesh(f_topo)
            st.session_state.bounds_design = get_mesh_bounds(st.session_state.mesh_design)
            st.session_state.bounds_topo = get_mesh_bounds(st.session_state.mesh_topo)
        
        os.unlink(f_design); os.unlink(f_topo)
        
        col1, col2 = st.columns(2)
        with col1:
            bd = st.session_state.bounds_design
            st.success(f"✅ Diseño cargado: {bd['n_faces']:,} caras, {bd['n_vertices']:,} vértices")
            st.caption(f"X: [{bd['xmin']:.1f}, {bd['xmax']:.1f}] | Y: [{bd['ymin']:.1f}, {bd['ymax']:.1f}] | Z: [{bd['zmin']:.1f}, {bd['zmax']:.1f}]")
        with col2:
            bt = st.session_state.bounds_topo
            st.success(f"✅ Topografía cargada: {bt['n_faces']:,} caras, {bt['n_vertices']:,} vértices")
            st.caption(f"X: [{bt['xmin']:.1f}, {bt['xmax']:.1f}] | Y: [{bt['ymin']:.1f}, {bt['ymax']:.1f}] | Z: [{bt['zmin']:.1f}, {bt['zmax']:.1f}]")
        
        st.session_state.step = 2
    except Exception as e:
        st.error(f"Error al cargar: {e}")

# =====================================================
# VISUALIZACIÓN 3D
# =====================================================
if st.session_state.mesh_design is not None and st.session_state.mesh_topo is not None:
    with st.expander("🌐 Vista 3D de Superficies", expanded=False):
        with st.spinner("Generando vista 3D..."):
            fig = go.Figure()
            
            md = decimate_mesh(st.session_state.mesh_design, 30000)
            mt = decimate_mesh(st.session_state.mesh_topo, 30000)
            
            fig.add_trace(mesh_to_plotly(md, "Diseño", "royalblue", 0.5))
            fig.add_trace(mesh_to_plotly(mt, "Topografía Real", "forestgreen", 0.5))
            
            # Dibujar secciones si existen
            if st.session_state.sections:
                for sec in st.session_state.sections:
                    from core.section_cutter import azimuth_to_direction
                    d = azimuth_to_direction(sec.azimuth)
                    p1 = sec.origin - d * sec.length / 2
                    p2 = sec.origin + d * sec.length / 2
                    bd = st.session_state.bounds_design
                    zmin, zmax = bd['zmin'], bd['zmax']
                    fig.add_trace(go.Scatter3d(
                        x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[(zmin+zmax)/2]*2,
                        mode='lines+text', text=[sec.name, ""],
                        line=dict(color='red', width=5),
                        name=sec.name, showlegend=False,
                    ))
            
            fig.update_layout(
                scene=dict(aspectmode='data',
                    xaxis_title='Este (m)', yaxis_title='Norte (m)', zaxis_title='Elevación (m)'),
                height=600, margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PASO 2: DEFINIR SECCIONES
# =====================================================
if st.session_state.step >= 2:
    st.header("✂️ Paso 2: Definir Secciones de Corte")
    
    tab_manual, tab_auto = st.tabs(["📌 Definición Manual", "🔄 Generación Automática"])
    
    with tab_manual:
        st.markdown("Define cada sección con un punto de origen (X, Y), azimut y longitud.")
        
        n_sections = st.number_input("Número de secciones a definir", min_value=1, max_value=50, value=5)
        
        sections_manual = []
        for i in range(n_sections):
            with st.expander(f"Sección S-{i+1:02d}", expanded=(i==0)):
                cols = st.columns(5)
                name = cols[0].text_input("Nombre", f"S-{i+1:02d}", key=f"sname_{i}")
                sector = cols[1].text_input("Sector", "", key=f"ssector_{i}")
                
                bd = st.session_state.bounds_design
                cx, cy = bd['center'][0], bd['center'][1]
                
                cols2 = st.columns(4)
                ox = cols2[0].number_input("Origen X", value=float(cx), format="%.1f", key=f"sox_{i}")
                oy = cols2[1].number_input("Origen Y", value=float(cy), format="%.1f", key=f"soy_{i}")
                az = cols2[2].number_input("Azimut (°)", value=0.0, min_value=0.0, max_value=360.0, key=f"saz_{i}")
                length = cols2[3].number_input("Longitud (m)", value=200.0, min_value=10.0, key=f"slen_{i}")
                
                sections_manual.append(SectionLine(name=name, origin=np.array([ox, oy]),
                    azimuth=az, length=length, sector=sector))
        
        if st.button("✅ Aplicar Secciones Manuales", type="primary"):
            st.session_state.sections = sections_manual
            st.session_state.step = 3
            st.success(f"✅ {len(sections_manual)} secciones definidas")
    
    with tab_auto:
        st.markdown("Genera secciones equiespaciadas a lo largo de una línea (ej: cresta del pit).")
        
        bd = st.session_state.bounds_design
        cols = st.columns(4)
        x1 = cols[0].number_input("Punto inicio X", value=float(bd['xmin']), format="%.1f")
        y1 = cols[1].number_input("Punto inicio Y", value=float(bd['center'][1]), format="%.1f")
        x2 = cols[2].number_input("Punto fin X", value=float(bd['xmax']), format="%.1f")
        y2 = cols[3].number_input("Punto fin Y", value=float(bd['center'][1]), format="%.1f")
        
        cols2 = st.columns(3)
        n_auto = cols2[0].number_input("N° de secciones", min_value=2, max_value=50, value=5)
        az_auto = cols2[1].number_input("Azimut de corte (°)", value=0.0, min_value=0.0, max_value=360.0)
        len_auto = cols2[2].number_input("Longitud de sección (m)", value=200.0, min_value=10.0)
        sector_auto = st.text_input("Nombre del sector", "Sector Principal")
        
        if st.button("🔄 Generar Secciones Automáticas", type="primary"):
            from core.section_cutter import generate_sections_along_crest
            sections_auto = generate_sections_along_crest(
                st.session_state.mesh_design,
                np.array([x1, y1]),
                np.array([x2, y2]),
                n_auto, az_auto, len_auto, sector_auto
            )
            st.session_state.sections = sections_auto
            st.session_state.step = 3
            st.success(f"✅ {len(sections_auto)} secciones generadas")
    
    # Mostrar tabla de secciones
    if st.session_state.sections:
        st.subheader("📋 Secciones Definidas")
        sec_data = []
        for s in st.session_state.sections:
            sec_data.append({
                "Nombre": s.name, "Sector": s.sector,
                "Origen X": f"{s.origin[0]:.1f}", "Origen Y": f"{s.origin[1]:.1f}",
                "Azimut (°)": f"{s.azimuth:.1f}", "Longitud (m)": f"{s.length:.1f}"
            })
        st.dataframe(sec_data, use_container_width=True)

# =====================================================
# PASO 3: CORTAR Y EXTRAER
# =====================================================
if st.session_state.step >= 3 and st.session_state.sections:
    st.header("🔬 Paso 3: Cortar Superficies y Extraer Parámetros")
    
    if st.button("🚀 Ejecutar Análisis", type="primary"):
        progress = st.progress(0)
        status = st.empty()
        
        profiles_d = []
        profiles_t = []
        params_d = []
        params_t = []
        comparisons = []
        
        total = len(st.session_state.sections)
        
        for i, section in enumerate(st.session_state.sections):
            status.text(f"Procesando sección {section.name} ({i+1}/{total})...")
            progress.progress((i + 1) / total)
            
            # Cortar ambas superficies
            pd = cut_mesh_with_section(st.session_state.mesh_design, section)
            pt = cut_mesh_with_section(st.session_state.mesh_topo, section)
            
            profiles_d.append(pd)
            profiles_t.append(pt)
            
            if pd is not None and pt is not None:
                # Extraer parámetros
                ep_d = extract_parameters(pd.distances, pd.elevations,
                    section.name, section.sector, resolution, face_threshold, berm_threshold)
                ep_t = extract_parameters(pt.distances, pt.elevations,
                    section.name, section.sector, resolution, face_threshold, berm_threshold)
                
                params_d.append(ep_d)
                params_t.append(ep_t)
                
                # Comparar
                if ep_d.benches and ep_t.benches:
                    comp = compare_design_vs_asbuilt(ep_d, ep_t, tolerances)
                    comparisons.extend(comp)
        
        st.session_state.profiles_design = profiles_d
        st.session_state.profiles_topo = profiles_t
        st.session_state.params_design = params_d
        st.session_state.params_topo = params_t
        st.session_state.comparison_results = comparisons
        st.session_state.step = 4
        
        status.text("✅ Análisis completado")
        
        # Resumen rápido
        n_ok = sum(1 for c in comparisons for k in ['height_status','angle_status','berm_status'] if c[k] == "CUMPLE")
        n_total = len(comparisons) * 3
        pct = n_ok / n_total * 100 if n_total > 0 else 0
        
        cols = st.columns(4)
        cols[0].metric("Secciones procesadas", f"{sum(1 for p in profiles_d if p is not None)}/{total}")
        cols[1].metric("Bancos detectados", len(comparisons))
        cols[2].metric("Total evaluaciones", n_total)
        cols[3].metric("Cumplimiento global", f"{pct:.1f}%")

# =====================================================
# PASO 4: RESULTADOS
# =====================================================
if st.session_state.step >= 4 and st.session_state.comparison_results:
    st.header("📊 Paso 4: Resultados")
    
    tab_profiles, tab_table, tab_dash, tab_export = st.tabs([
        "📈 Perfiles", "📋 Tabla Detallada", "📊 Dashboard", "💾 Exportar"
    ])
    
    # --- PERFILES ---
    with tab_profiles:
        for i, section in enumerate(st.session_state.sections):
            pd = st.session_state.profiles_design[i]
            pt = st.session_state.profiles_topo[i]
            
            if pd is None or pt is None:
                st.warning(f"⚠️ Sección {section.name}: sin intersección con una o ambas superficies")
                continue
            
            fig = make_subplots(rows=1, cols=1)
            fig.add_trace(go.Scatter(x=pd.distances, y=pd.elevations,
                mode='lines', name='Diseño', line=dict(color='royalblue', width=2)))
            fig.add_trace(go.Scatter(x=pt.distances, y=pt.elevations,
                mode='lines', name='Topografía Real', line=dict(color='forestgreen', width=2)))
            
            # Marcar bancos detectados
            if i < len(st.session_state.params_topo):
                for bench in st.session_state.params_topo[i].benches:
                    fig.add_annotation(x=bench.crest_distance, y=bench.crest_elevation,
                        text=f"B{bench.bench_number}", showarrow=True, arrowhead=2,
                        font=dict(size=10, color="red"))
            
            fig.update_layout(
                title=f"Sección {section.name} — {section.sector}",
                xaxis_title="Distancia (m)", yaxis_title="Elevación (m)",
                height=400, yaxis=dict(scaleanchor="x", scaleratio=1),
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                margin=dict(l=60, r=20, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # --- TABLA ---
    with tab_table:
        import pandas as pd
        
        if st.session_state.comparison_results:
            df = pd.DataFrame(st.session_state.comparison_results)
            
            display_cols = {
                'sector': 'Sector', 'section': 'Sección', 'bench_num': 'Banco',
                'level': 'Nivel', 'height_design': 'H. Diseño', 'height_real': 'H. Real',
                'height_dev': 'Desv. H', 'height_status': 'Cumpl. H',
                'angle_design': 'Á. Diseño', 'angle_real': 'Á. Real',
                'angle_dev': 'Desv. Á', 'angle_status': 'Cumpl. Á',
                'berm_design': 'B. Diseño', 'berm_real': 'B. Real',
                'berm_dev': 'Desv. B', 'berm_status': 'Cumpl. B',
            }
            df_display = df.rename(columns=display_cols)
            
            def highlight_status(val):
                if val == "CUMPLE": return 'background-color: #C6EFCE; color: #006100'
                elif val == "FUERA DE TOLERANCIA": return 'background-color: #FFEB9C; color: #9C5700'
                elif val == "NO CUMPLE": return 'background-color: #FFC7CE; color: #9C0006'
                return ''
            
            styled = df_display.style.map(highlight_status,
                subset=['Cumpl. H', 'Cumpl. Á', 'Cumpl. B'])
            st.dataframe(styled, use_container_width=True, height=400)
    
    # --- DASHBOARD ---
    with tab_dash:
        results = st.session_state.comparison_results
        
        # Métricas por parámetro
        cols = st.columns(3)
        for col, (param, key, label) in zip(cols, [
            ('height', 'height_status', 'Altura de Banco'),
            ('angle', 'angle_status', 'Ángulo de Cara'),
            ('berm', 'berm_status', 'Ancho de Berma'),
        ]):
            total = len(results)
            cumple = sum(1 for r in results if r[key] == "CUMPLE")
            pct = cumple / total * 100 if total > 0 else 0
            col.metric(label, f"{pct:.0f}%", f"{cumple}/{total} cumplen")
        
        # Gráfico de barras de cumplimiento
        status_counts = {'Parámetro': [], 'CUMPLE': [], 'FUERA DE TOLERANCIA': [], 'NO CUMPLE': []}
        for key, label in [('height_status','Altura'), ('angle_status','Ángulo Cara'), ('berm_status','Berma')]:
            status_counts['Parámetro'].append(label)
            status_counts['CUMPLE'].append(sum(1 for r in results if r[key] == "CUMPLE"))
            status_counts['FUERA DE TOLERANCIA'].append(sum(1 for r in results if r[key] == "FUERA DE TOLERANCIA"))
            status_counts['NO CUMPLE'].append(sum(1 for r in results if r[key] == "NO CUMPLE"))
        
        import pandas as pd
        df_status = pd.DataFrame(status_counts)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name='CUMPLE', x=df_status['Parámetro'], y=df_status['CUMPLE'],
            marker_color='#006100'))
        fig_bar.add_trace(go.Bar(name='FUERA TOL.', x=df_status['Parámetro'], y=df_status['FUERA DE TOLERANCIA'],
            marker_color='#9C5700'))
        fig_bar.add_trace(go.Bar(name='NO CUMPLE', x=df_status['Parámetro'], y=df_status['NO CUMPLE'],
            marker_color='#9C0006'))
        fig_bar.update_layout(barmode='stack', title="Cumplimiento por Parámetro",
            height=350, margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Distribución de desviaciones
        col1, col2 = st.columns(2)
        with col1:
            devs_h = [r['height_dev'] for r in results]
            fig_h = go.Figure(go.Histogram(x=devs_h, nbinsx=15, marker_color='royalblue'))
            fig_h.update_layout(title="Distribución Desv. Altura (m)", height=300,
                xaxis_title="Desviación (m)", yaxis_title="Frecuencia")
            fig_h.add_vline(x=-tol_h_neg, line_dash="dash", line_color="orange")
            fig_h.add_vline(x=tol_h_pos, line_dash="dash", line_color="orange")
            st.plotly_chart(fig_h, use_container_width=True)
        with col2:
            devs_a = [r['angle_dev'] for r in results]
            fig_a = go.Figure(go.Histogram(x=devs_a, nbinsx=15, marker_color='forestgreen'))
            fig_a.update_layout(title="Distribución Desv. Ángulo Cara (°)", height=300,
                xaxis_title="Desviación (°)", yaxis_title="Frecuencia")
            fig_a.add_vline(x=-tol_a_neg, line_dash="dash", line_color="orange")
            fig_a.add_vline(x=tol_a_pos, line_dash="dash", line_color="orange")
            st.plotly_chart(fig_a, use_container_width=True)
    
    # --- EXPORTAR ---
    with tab_export:
        st.subheader("💾 Exportar Resultados a Excel")
        
        if st.button("📥 Generar Excel de Conciliación", type="primary"):
            with st.spinner("Generando Excel..."):
                output_path = os.path.join(tempfile.gettempdir(), "Conciliacion_Resultados.xlsx")
                project_info = {
                    'project': project_name, 'operation': operation,
                    'phase': phase, 'author': author,
                    'date': datetime.now().strftime("%d/%m/%Y"),
                }
                export_results(
                    st.session_state.comparison_results,
                    st.session_state.params_design,
                    st.session_state.params_topo,
                    tolerances, output_path, project_info
                )
                
                with open(output_path, "rb") as f:
                    st.download_button(
                        "⬇️ Descargar Excel",
                        f.read(),
                        file_name="Conciliacion_Diseno_vs_AsBuilt.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                    )
            st.success("✅ Excel generado exitosamente")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    Conciliación Geotécnica v1.0 | Herramienta de análisis Diseño vs As-Built<br>
    Parámetros: Banco 15m | Cara 65°-75° | Berma 8-10m
</div>
""", unsafe_allow_html=True)
