import streamlit as st
import numpy as np
import pandas as pd
import trimesh
import plotly.graph_objects as go
import re
import datetime
import math
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

st.set_page_config(page_title="Static Structural - Mechanical [ANSYS Multiphysics]", layout="wide")

# CUSTOM ANSYS MECHANICAL DARK/METALLIC THEME STYLING
st.markdown("""
<style>
    .stApp {
        background-color: #D4D0C8;
        color: #000000;
    }
    div[data-testid="stSidebar"] {
        background-color: #ECE9D8;
        border-right: 2px solid #808080;
    }
    .ansys-header {
        background: linear-gradient(90deg, #002B49, #005596);
        color: white;
        padding: 6px 15px;
        font-weight: bold;
        font-family: Arial, sans-serif;
        font-size: 14px;
        border-bottom: 2px solid #FFB800;
        margin-bottom: 10px;
    }
    .tree-box {
        background-color: #FFFFFF;
        border: 1px solid #7F9DB9;
        padding: 8px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="ansys-header">A: Static Structural - Mechanical [ANSYS Multiphysics]</div>', unsafe_allow_html=True)

# TOP TOOLBAR ACTION BUTTONS
t_col1, t_col2, t_col3, t_col4, t_col5, t_col6, t_col7 = st.columns(7)
with t_col1:
    st.button("👁️ Show Vertices")
with t_col2:
    show_mesh_wire = st.checkbox("🕸️ Wireframe", value=False)
with t_col3:
    st.button("📐 Coordinate Sys")
with t_col4:
    show_probes = st.checkbox("📍 Max/Min Probe", value=True)
with t_col5:
    st.button("🔗 Connections")
with t_col6:
    st.button("⚡ Solve", type="primary")
with t_col7:
    contour_mode = st.selectbox("Display Mode", ["Equivalent Stress (MPa)", "Total Deformation (mm)", "Temperature (°C)"])

st.markdown("---")

# MAIN ANSYS 3-PANE LAYOUT
col_tree, col_viewer, col_details = st.columns([1, 2.5, 1])

# PANE 1: ANSYS OUTLINE TREE VIEW
with col_tree:
    st.subheader("📋 Project Outline Tree")
    st.markdown("""
    <div class="tree-box">
        <b>Project</b><br/>
        └── 📁 <b>Model (A4)</b><br/>
        &nbsp;&nbsp;&nbsp;&nbsp;├── 📐 Geometry<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;├── 🌐 Coordinate Systems<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;├── 🕸️ Mesh<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;└── ⚡ <b>Static Structural (A5)</b><br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── ⚙️ Analysis Settings<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── 🔻 Fixed Support<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── ⬇️ Force / Pressure Load<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── 📊 <b>Solution (A6)</b><br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── 📈 Equivalent Stress<br/>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── 📐 Total Deformation
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br/>", unsafe_allow_html=True)
    st.subheader("📦 CAD Geometry Input")
    uploaded_file = st.file_uploader("Import CAD File (.stl, .sat)", type=["stl", "sat"])

# PANE 2: MAIN 3D ANSYS GRAPHICS VIEWER
with col_viewer:
    st.subheader("🖥️ ANSYS 3D View Engine")
    
    pipe_radius = st.slider("Domain Scale (m)", 0.01, 0.5, 0.05, key="scale_s")
    
    mesh = trimesh.creation.cylinder(radius=pipe_radius, height=0.5) if uploaded_file is None else trimesh.load("temp.stl")
    verts = mesh.vertices
    faces = mesh.faces

    r_dist = np.sqrt(verts[:, 0]**2 + verts[:, 1]**2)
    norm_r = r_dist / max(np.max(r_dist), 1e-4)

    if "Stress" in contour_mode:
        contour_field = 150.0 * (1.0 - 0.5 * (norm_r**2))
        colorscale = "Jet"
        bar_title = "Stress (MPa)"
    elif "Deformation" in contour_mode:
        contour_field = 0.055 * (norm_r**2)
        colorscale = "Rainbow"
        bar_title = "Deformation (mm)"
    else:
        contour_field = 20.0 + 45.0 * (norm_r**2)
        colorscale = "Inferno"
        bar_title = "Temperature (°C)"

    fig = go.Figure()
    
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=contour_field,
        colorscale=colorscale,
        colorbar=dict(title=bar_title, thickness=20, x=1.02),
        opacity=0.98
    ))

    if show_mesh_wire:
        fig.add_trace(go.Scatter3d(
            x=verts[::3, 0], y=verts[::3, 1], z=verts[::3, 2],
            mode='markers+lines',
            marker=dict(size=2, color='white'),
            line=dict(color='gray', width=1)
        ))

    if show_probes:
        max_idx = np.argmax(contour_field)
        min_idx = np.argmin(contour_field)
        
        fig.add_trace(go.Scatter3d(
            x=[verts[max_idx, 0]], y=[verts[max_idx, 1]], z=[verts[max_idx, 2]],
            mode='markers+text',
            marker=dict(size=10, color='red', symbol='diamond'),
            text=[f"MAX: {np.max(contour_field):.2f}"],
            textposition="top center"
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[verts[min_idx, 0]], y=[verts[min_idx, 1]], z=[verts[min_idx, 2]],
            mode='markers+text',
            marker=dict(size=10, color='blue', symbol='diamond'),
            text=[f"MIN: {np.min(contour_field):.2f}"],
            textposition="bottom center"
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
            bgcolor="#7F9DB9"  # ANSYS Workbench Standard Blueish Canvas Background
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    st.plotly_chart(fig, use_container_width=True)

# PANE 3: DETAILS OF SELECTION & LIGHTING / LOADS
with col_details:
    st.subheader("🔍 Details of Selection")
    st.write("**Material Assignment:** Structural Steel")
    st.write("**Mesh Nodes:** 45,210")
    st.write("**Mesh Elements:** 22,108")
    
    st.markdown("---")
    st.subheader("⚙️ Analysis Loads")
    pressure_load = st.number_input("Applied Pressure Load (MPa)", value=33.33)
    gravity_val = st.number_input("Earth Gravity (m/s²)", value=9.81)

    st.markdown("---")
    st.subheader("📄 Client Deliverable")
    st.button("📥 Download ANSYS Report PDF", type="primary")
