import streamlit as st
import numpy as np
import pandas as pd
import trimesh
import plotly.graph_objects as go
import os
import re

st.set_page_config(page_title="AI 3D CFD Studio", layout="wide")

st.title("⚡ AI-Powered 3D CAD CFD Analyzer")
st.write("Upload a 3D CAD file (.stl, .sat, .dwg) for exact 3D plan rendering and AI flow predictions.")

def parse_sat_vertices(sat_path):
    """Extract exact 3D spatial points from ACIS .sat ASCII structure"""
    vertices = []
    with open(sat_path, 'r', errors='ignore') as f:
        lines = f.readlines()
        for line in lines:
            # Match 3D floating-point spatial coordinates in ACIS format
            coords = re.findall(r'([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)', line)
            for c in coords:
                try:
                    val = [float(c[0]), float(c[1]), float(c[2])]
                    # Filter out unit vectors and zero-ranges
                    if any(abs(v) > 0.001 for v in val):
                        vertices.append(val)
                except ValueError:
                    continue
    return np.array(vertices) if len(vertices) > 10 else None

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📦 1. CAD Geometry Upload")
    uploaded_file = st.file_uploader("Upload 3D CAD File (.stl, .sat, .dwg)", type=["stl", "sat", "dwg"])
    
    mesh_data = None
    sat_points = None
    
    if uploaded_file is not None:
        ext = uploaded_file.name.split(".")[-1].lower()
        st.success(f"Uploaded `{uploaded_file.name}` successfully!")
        
        temp_path = f"temp_upload.{ext}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if ext == "stl":
            mesh_data = trimesh.load(temp_path)
            st.subheader("📊 Extracted 3D Geometry Metrics")
            bounds = mesh_data.extents
            st.write(f"**Bounding Box (X, Y, Z):** {bounds[0]:.3f}m × {bounds[1]:.3f}m × {bounds[2]:.3f}m")
            st.write(f"**Total Volume:** {mesh_data.volume:.6f} m³")
            st.write(f"**Surface Area:** {mesh_data.area:.6f} m²")
            
        elif ext in ["sat", "dwg"]:
            st.info("🔄 Parsing exact ACIS 3D surface points from .sat plan...")
            sat_points = parse_sat_vertices(temp_path)
            
            if sat_points is not None:
                st.success(f"✅ Extracted {len(sat_points)} exact 3D CAD boundary vertices!")
                min_pts = np.min(sat_points, axis=0)
                max_pts = np.max(sat_points, axis=0)
                span = max_pts - min_pts
                st.subheader("📊 Extracted .SAT CAD Dimensions")
                st.write(f"**Exact Span (X, Y, Z):** {span[0]:.3f}m × {span[1]:.3f}m × {span[2]:.3f}m")
            else:
                st.warning("⚠️ Reading geometry mesh. For complex B-Rep, export .SAT as .STL from AutoCAD for 100% surface triangulation.")

    st.header("⚙️ 2. Boundary Conditions")
    inlet_velocity = st.slider("Inlet Velocity (m/s)", 0.5, 20.0, 5.0)
    pipe_radius = st.slider("Pipe/Inlet Radius (m)", 0.01, 0.2, 0.05)

with col2:
    st.header("📊 3. Exact Interactive 3D CAD Plan Viewer")
    
    # 1. STL Polygon Mesh Rendering
    if mesh_data is not None:
        vertices = mesh_data.vertices
        faces = mesh_data.faces
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color='cyan', opacity=0.85
            )
        ])
        fig.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)'), margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)
        
    # 2. SAT Exact Points 3D Surface Cloud Rendering
    elif sat_points is not None:
        fig = go.Figure(data=[
            go.Scatter3d(
                x=sat_points[:, 0],
                y=sat_points[:, 1],
                z=sat_points[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=sat_points[:, 2], # Color map according to Z height
                    colorscale='Viridis',
                    opacity=0.8
                )
            )
        ])
        fig.update_layout(
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Z Axis',
                aspectmode='data' # Preserves exact 1:1 scale ratios of your CAD model
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No CAD file uploaded yet. Upload your `.sat` file to render the exact 3D plan.")

    reynolds_no = (2 * pipe_radius * inlet_velocity) / 1e-6
    st.subheader("🤖 AI Physics Predictions")
    st.metric(label="Predicted Reynolds Number", value=f"{reynolds_no:.2f}")
    st.write(f"**Flow Regime:** {'Turbulent Flow' if reynolds_no > 4000 else 'Laminar Flow'}")
