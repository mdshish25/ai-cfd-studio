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

# Streamlit Page Setup
st.set_page_config(page_title="ANSYS Multi-Physics & AI Workstation 2026", layout="wide", initial_sidebar_state="expanded")

# CUSTOM WORKSTATION THEME & CHATBOT STYLING
st.markdown("""
<style>
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Segoe UI', Inter, sans-serif;
    }
    .ansys-top-banner {
        background: linear-gradient(90deg, #0284C7, #0F172A);
        color: #FFFFFF;
        padding: 10px 20px;
        font-weight: 700;
        font-size: 16px;
        letter-spacing: 0.5px;
        border-bottom: 2px solid #38BDF8;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    .ansys-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .card-title {
        color: #38BDF8;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 12px;
        border-bottom: 1px solid #334155;
        padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MATERIAL LIBRARY DATABASE
# ---------------------------------------------------------
MATERIALS_DB = {
    "Air (Ideal Gas)": {"density": 1.225, "viscosity": 1.81e-5, "k": 0.026, "cp": 1005},
    "Water (Liquid)": {"density": 998.0, "viscosity": 1.005e-3, "k": 0.6, "cp": 4182},
    "Liquid Methane (CH4)": {"density": 422.0, "viscosity": 1.1e-4, "k": 0.19, "cp": 3480},
    "Structural Steel": {"density": 7850, "viscosity": 0.0, "k": 60.5, "cp": 434},
    "Aluminum 6061-T6": {"density": 2700, "viscosity": 0.0, "k": 167.0, "cp": 896},
    "Titanium Grade 5": {"density": 4430, "viscosity": 0.0, "k": 6.7, "cp": 526}
}

# ---------------------------------------------------------
# SIDEBAR: AI ASSISTANT & AUTO DEBUGGER PANEL
# ---------------------------------------------------------
with st.sidebar:
    st.header("🤖 ANSYS AI Engineering Assistant")
    st.caption("Auto-Code Debugger & Physics Expert Engine")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Namaste! Main aapka **ANSYS Multi-Physics AI Assistant** hoon. Combustion, VOF Multiphase, FEA/CFD, ya kisi bhi Streamlit code error ka solution poochne ke liye query type karein."}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_input := st.chat_input("Poochhein (e.g. Combustion temperature / VOF phase ka formula)..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        query_lower = user_input.lower()
        if "combustion" in query_lower or "flame" in query_lower:
            reply = "Combustion Module me $CH_4 + 2O_2 \\rightarrow CO_2 + 2H_2O$ reaction kinetics solve hoti hai. Isse adiabatic flame temperature ($T_{flame} \\sim 2220\\,K$) compute hota hai."
        elif "vof" in query_lower or "multiphase" in query_lower or "phase" in query_lower:
            reply = "Multiphase VOF (Volume of Fluid) Module $\\alpha_{phase} \\in [0, 1]$ indicator equation se Primary (e.g. Air) aur Secondary (e.g. Water) phase ke interface ko track karta hai."
        elif "material" in query_lower:
            reply = "Aap **Custom Material Library** se Air, Water, Liquid Methane, Structural Steel, Aluminum, ya Titanium select kar sakte hain."
        elif "error" in query_lower or "bug" in query_lower or "syntax" in query_lower:
            reply = "Agar koi error aa raha hai, toh exact error log paste karein. Main code ka fixed patch instantly generate kar doonga."
        else:
            reply = f"Aapne poocha: '{user_input}'. Main ANSYS Mechanical, Fluent CFD, Combustion, Multiphase VOF, aur Material Library ke multi-physics equations ke hisaab se support kar sakta hoon."

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

# ---------------------------------------------------------
# MAIN WORKSTATION DASHBOARD
# ---------------------------------------------------------
st.markdown("""
<div class="ansys-top-banner">
    <div>⚡ ANSYS Discovery 2026 R1 - Advanced Multi-Physics AI Studio</div>
    <div style="font-size: 12px; opacity: 0.8;">CFD | FEA | Combustion | Multiphase VOF</div>
</div>
""", unsafe_allow_html=True)

tb_col1, tb_col2, tb_col3, tb_col4 = st.columns([1, 1, 1.2, 1.5])
with tb_col1:
    show_mesh_wire = st.checkbox("🕸️ Mesh Wireframe", value=False)
with tb_col2:
    show_probes = st.checkbox("📍 Sensor Probes", value=True)
with tb_col3:
    physics_mode = st.selectbox("Physics Module", ["CFD Fluid Dynamics", "Multiphase VOF", "Combustion Analysis", "Static Structural FEA"])
with tb_col4:
    uploaded_file = st.file_uploader("Upload CAD (.stl, .sat)", type=["stl", "sat"], label_visibility="collapsed")

col_viewer, col_details = st.columns([3.2, 1.2])

mesh = None
filename_str = "CAD_Model.stl"

def parse_sat_file(sat_path):
    raw_vertices = []
    with open(sat_path, 'r', errors='ignore') as f:
        content = f.read()
        pattern = r'([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)'
        matches = re.findall(pattern, content)
        for m in matches:
            try:
                v = [float(m[0]), float(m[1]), float(m[2])]
                if any(abs(c) > 0.0001 for c in v) and all(abs(c) < 2000.0 for c in v):
                    raw_vertices.append(v)
            except ValueError:
                continue

    if len(raw_vertices) < 12:
        return None

    vertices = np.unique(np.array(raw_vertices), axis=0)
    try:
        cloud = trimesh.PointCloud(vertices)
        mesh = trimesh.convex.convex_hull(cloud)
        return mesh
    except Exception:
        return None

if uploaded_file is not None:
    filename_str = uploaded_file.name
    ext = filename_str.split(".")[-1].lower()
    temp_path = f"temp_upload.{ext}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        if ext == "stl":
            mesh = trimesh.load(temp_path, force='mesh')
            if isinstance(mesh, trimesh.Scene):
                geometries = list(mesh.geometry.values())
                if len(geometries) > 0:
                    mesh = trimesh.util.concatenate(geometries)
        elif ext == "sat":
            mesh = parse_sat_file(temp_path)
    except Exception:
        mesh = None

if mesh is None or not isinstance(mesh, trimesh.Trimesh):
    mesh = trimesh.creation.cylinder(radius=0.05, height=0.5)

verts = mesh.vertices
faces = mesh.faces

r_dist = np.sqrt(verts[:, 0]**2 + verts[:, 1]**2)
norm_r = r_dist / max(np.max(r_dist), 1e-5)
z_coords = verts[:, 2]
norm_z = (z_coords - np.min(z_coords)) / max(np.ptp(z_coords), 1e-5)

with col_details:
    st.markdown('<div class="ansys-card"><div class="card-title">🧱 Material & Physics Setup</div>', unsafe_allow_html=True)
    selected_mat = st.selectbox("Assign Engineering Material:", list(MATERIALS_DB.keys()))
    mat_props = MATERIALS_DB[selected_mat]
    st.caption(f"Density: `{mat_props['density']}` kg/m³ | k: `{mat_props['k']}` W/m·K")

    if physics_mode == "CFD Fluid Dynamics":
        inlet_velocity = st.slider("Inlet Flow Velocity V_in (m/s)", 0.5, 50.0, 10.0)
        dh = 2 * 0.05
        reynolds_no = (mat_props['density'] * inlet_velocity * dh) / max(mat_props['viscosity'], 1e-6)
        dynamic_pressure = 0.5 * mat_props['density'] * (inlet_velocity**2)
        
        contour_field = inlet_velocity * (1.0 - 0.75 * (norm_r**2))
        colorscale = "Jet"
        bar_title = "Velocity (m/s)"

        st.markdown('</div><div class="ansys-card"><div class="card-title">📊 CFD Metrics</div>', unsafe_allow_html=True)
        st.metric("Reynolds Number (Re)", f"{reynolds_no:,.0f}")
        st.metric("Dynamic Pressure", f"{dynamic_pressure:.2f} Pa")
        st.metric("Max Flow Velocity", f"{np.max(contour_field):.2f} m/s")
        st.markdown('</div>', unsafe_allow_html=True)

    elif physics_mode == "Multiphase VOF":
        fill_ratio = st.slider("Primary Fluid Volume Fraction (α)", 0.0, 1.0, 0.45)
        contour_field = np.where(norm_z <= fill_ratio, 1.0, 0.0)
        colorscale = "Blues"
        bar_title = "Water Phase Fraction (α)"

        st.markdown('</div><div class="ansys-card"><div class="card-title">🌊 VOF Metrics</div>', unsafe_allow_html=True)
        st.metric("Phase 1 (Liquid)", f"{fill_ratio * 100:.1f} %")
        st.metric("Phase 2 (Gas)", f"{(1 - fill_ratio) * 100:.1f} %")
        st.metric("Interfacial Tension", "0.072 N/m")
        st.markdown('</div>', unsafe_allow_html=True)

    elif physics_mode == "Combustion Analysis":
        equivalence_ratio = st.slider("Equivalence Ratio (Φ)", 0.5, 1.5, 1.0)
        t_flame = 300.0 + (1920.0 * (1.0 - abs(equivalence_ratio - 1.0) * 0.5))
        contour_field = 300.0 + (t_flame - 300.0) * (1.0 - norm_r**2) * (norm_z)
        colorscale = "Hot"
        bar_title = "Temperature (K)"

        st.markdown('</div><div class="ansys-card"><div class="card-title">🔥 Flame Metrics</div>', unsafe_allow_html=True)
        st.metric("Peak Flame Temperature", f"{np.max(contour_field):.1f} K")
        st.metric("CO2 Species Mass Frac", "0.142")
        st.metric("H2O Species Mass Frac", "0.116")
        st.markdown('</div>', unsafe_allow_html=True)

    else: # Static Structural FEA
        applied_load = st.number_input("Applied Load (MPa)", value=33.33)
        contour_field = applied_load * 12.5 * (1.0 - 0.45 * norm_r**2)
        colorscale = "Rainbow"
        bar_title = "Stress (MPa)"

        st.markdown('</div><div class="ansys-card"><div class="card-title">⚙️ FEA Metrics</div>', unsafe_allow_html=True)
        st.metric("Max von-Mises Stress", f"{np.max(contour_field):.2f} MPa")
        st.metric("Safety Factor", f"{(250.0 / np.max(contour_field)):.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

with col_viewer:
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=contour_field,
        colorscale=colorscale,
        colorbar=dict(title=bar_title, thickness=18, x=1.01, len=0.8, tickfont=dict(color='white')),
        opacity=0.98,
        lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.1)
    ))

    if show_mesh_wire:
        fig.add_trace(go.Scatter3d(
            x=verts[::3, 0], y=verts[::3, 1], z=verts[::3, 2],
            mode='markers+lines',
            marker=dict(size=2, color='#38BDF8'),
            line=dict(color='#475569', width=1)
        ))

    if show_probes:
        max_idx = np.argmax(contour_field)
        min_idx = np.argmin(contour_field)
        
        fig.add_trace(go.Scatter3d(
            x=[verts[max_idx, 0]], y=[verts[max_idx, 1]], z=[verts[max_idx, 2]],
            mode='markers+text',
            marker=dict(size=9, color='#EF4444', symbol='diamond'),
            text=[f"MAX: {np.max(contour_field):.2f}"],
            textfont=dict(color='#EF4444', size=11),
            textposition="top center"
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (m)', backgroundcolor="#0F172A", gridcolor="#334155", showbackground=True),
            yaxis=dict(title='Y (m)', backgroundcolor="#0F172A", gridcolor="#334155", showbackground=True),
            zaxis=dict(title='Z (m)', backgroundcolor="#0F172A", gridcolor="#334155", showbackground=True),
            aspectmode='data'
        ),
        height=620,
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="#1E293B"
    )
    st.plotly_chart(fig, use_container_width=True)
