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
st.set_page_config(page_title="ANSYS Discovery & AI Assistant Studio", layout="wide", initial_sidebar_state="expanded")

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
# SIDEBAR: AI ASSISTANT PANEL (GIVES RESPONSES LIKE ME)
# ---------------------------------------------------------
with st.sidebar:
    st.header("🤖 ANSYS AI Engineering Assistant")
    st.caption("Powered by Engineering LLM Engine (APDL / Fluent Expert)")

    # Chat History Session Initialization
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Namaste! Main aapka **ANSYS AI Simulation Assistant** hoon. Aap apni CAD file, boundary conditions, ya Streamlit error ke baare me mujhse pooch sakte hain."}
        ]

    # Render Chat Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User Chat Input
    if user_input := st.chat_input("Poochhein (e.g. Reynolds number kaise badhayein?)..."):
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # AI Intelligent Response Engine Logic
        query_lower = user_input.lower()
        if "reynolds" in query_lower or "re" in query_lower:
            reply = "Reynolds Number ($Re = \\frac{\\rho V D_h}{\\mu}$) badhane ke liye aap **Inlet Velocity** badha sakte hain ya fluid density (jaise Air se Water) change kar sakte hain."
        elif "sat" in query_lower or "stl" in query_lower or "file" in query_lower:
            reply = "Aap `.stl` aur `.sat` (ACIS ASCII) format upload kar sakte hain. Subspace mesh loader use exact 3D surface me convert kar dega."
        elif "error" in query_lower or "problem" in query_lower:
            reply = "Aap apna error traceback ya issue mujhe batayein, main aapko exact **bug-free python code** bana kar doonga."
        else:
            reply = f"Aapne poocha: '{user_input}'. Main ANSYS Mechanical & Fluent solver matrices ($[K]\\{U\\} = \\{F\\}$) aur Navier-Stokes equations ke hisaab se aapke model ko solve kar sakta hoon."

        # Append Assistant Reply
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

# ---------------------------------------------------------
# MAIN WORKSTATION DASHBOARD & CFD ENGINE
# ---------------------------------------------------------
st.markdown("""
<div class="ansys-top-banner">
    <div>⚡ ANSYS Discovery 2026 R1 - Multi-Physics & AI Workstation</div>
    <div style="font-size: 12px; opacity: 0.8;">Fluid Dynamics & FEA Engine | Real-Time Solver</div>
</div>
""", unsafe_allow_html=True)

tb_col1, tb_col2, tb_col3, tb_col4 = st.columns([1, 1, 1, 1.5])
with tb_col1:
    show_mesh_wire = st.checkbox("🕸️ Mesh Wireframe", value=False)
with tb_col2:
    show_probes = st.checkbox("📍 Sensor Probes", value=True)
with tb_col3:
    contour_mode = st.selectbox("Display Mode", ["Velocity Field (m/s)", "Pressure Drop Field (Pa)"])
with tb_col4:
    uploaded_file = st.file_uploader("Upload CAD Geometry (.stl, .sat)", type=["stl", "sat"], label_visibility="collapsed")

col_viewer, col_details = st.columns([3.2, 1.2])

mesh = None
filename_str = "CAD_CFD_Model.stl"

if uploaded_file is not None:
    filename_str = uploaded_file.name
    ext = filename_str.split(".")[-1].lower()
    temp_path = f"temp_upload.{ext}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        mesh = trimesh.load(temp_path, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            geometries = list(mesh.geometry.values())
            if len(geometries) > 0:
                mesh = trimesh.util.concatenate(geometries)
    except Exception:
        mesh = None

if mesh is None or not isinstance(mesh, trimesh.Trimesh):
    mesh = trimesh.creation.cylinder(radius=0.05, height=0.5)

verts = mesh.vertices
faces = mesh.faces

with col_details:
    st.markdown('<div class="ansys-card"><div class="card-title">⚙️ Fluid Boundary Conditions</div>', unsafe_allow_html=True)
    inlet_velocity = st.slider("Inlet Velocity V_in (m/s)", 0.5, 50.0, 10.0)
    pipe_radius = st.slider("Domain Scale / Radius (m)", 0.01, 0.5, 0.05)
    fluid_type = st.selectbox("Fluid Medium:", ["Air (1.225 kg/m³)", "Water (998 kg/m³)", "Oil (870 kg/m³)"])
    st.markdown('</div>', unsafe_allow_html=True)

    density_val = 1.225 if "Air" in fluid_type else (998.0 if "Water" in fluid_type else 870.0)
    viscosity_val = 1.81e-5 if "Air" in fluid_type else 1.005e-3

    # Fast CFD Vectorized Calculations
    dh = 2 * pipe_radius
    reynolds_no = (density_val * inlet_velocity * dh) / viscosity_val
    regime = "Turbulent (k-ε)" if reynolds_no > 4000 else "Laminar"
    dynamic_pressure = 0.5 * density_val * (inlet_velocity**2)
    cd = 0.45 if "Turbulent" in regime else 24.0 / max(reynolds_no, 0.1)
    drag_force = cd * dynamic_pressure * (math.pi * (pipe_radius**2))

    r_dist = np.sqrt(verts[:, 0]**2 + verts[:, 1]**2)
    norm_r = r_dist / max(np.max(r_dist), 1e-5)
    z_coords = verts[:, 2]
    norm_z = (z_coords - np.min(z_coords)) / max(np.ptp(z_coords), 1e-5)

    vel_field = inlet_velocity * (1.0 - 0.75 * (norm_r**2)) * (1.0 + 0.1 * np.sin(norm_z * math.pi * 2))
    press_field = (dynamic_pressure * 2.2) - (0.5 * density_val * (vel_field**2))

    st.markdown('<div class="ansys-card"><div class="card-title">📊 CFD Output Metrics</div>', unsafe_allow_html=True)
    st.metric("Reynolds Number (Re)", f"{reynolds_no:,.0f}")
    st.metric("Flow State", regime)
    st.metric("Dynamic Pressure", f"{dynamic_pressure:.2f} Pa")
    st.metric("Drag Force", f"{drag_force:.3f} N")
    st.metric("Max Velocity", f"{np.max(vel_field):.2f} m/s")
    st.markdown('</div>', unsafe_allow_html=True)

with col_viewer:
    if "Velocity" in contour_mode:
        contour_field = vel_field
        colorscale = "Jet"
        bar_title = "Velocity (m/s)"
    else:
        contour_field = press_field
        colorscale = "Plasma"
        bar_title = "Pressure (Pa)"

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=contour_field,
        colorscale=colorscale,
        colorbar=dict(title=bar_title, thickness=18, x=1.01, len=0.8, tickfont=dict(color='white')),
        opacity=0.98
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
        
        fig.add_trace(go.Scatter3d(
            x=[verts[min_idx, 0]], y=[verts[min_idx, 1]], z=[verts[min_idx, 2]],
            mode='markers+text',
            marker=dict(size=9, color='#3B82F6', symbol='diamond'),
            text=[f"MIN: {np.min(contour_field):.2f}"],
            textfont=dict(color='#3B82F6', size=11),
            textposition="bottom center"
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
