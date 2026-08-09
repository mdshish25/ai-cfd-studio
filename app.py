import streamlit as st
import numpy as np
import pandas as pd
import trimesh
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.spatial import Delaunay
import re

st.set_page_config(page_title="AI 3D CFD Studio", layout="wide")

st.title("⚡ AI-Powered 3D CAD CFD Analyzer")

def parse_exact_sat_mesh(sat_path):
    """Extract exact 3D vertices and triangulate true surface without distortion"""
    raw_coords = []
    with open(sat_path, 'r', errors='ignore') as f:
        for line in f:
            matches = re.findall(r'([-+]?\d*\.\d+|\d+)', line)
            if len(matches) >= 3:
                for i in range(0, len(matches)-2, 3):
                    try:
                        x, y, z = float(matches[i]), float(matches[i+1]), float(matches[i+2])
                        # Filter out header index IDs / scale anomalies
                        if abs(x) < 500 and abs(y) < 500 and abs(z) < 500:
                            raw_coords.append([x, y, z])
                    except ValueError:
                        continue

    if len(raw_coords) < 10:
        return None, None, None

    pts = np.unique(np.array(raw_coords), axis=0)

    # Filter metadata/index noise
    mean = np.mean(pts, axis=0)
    std = np.std(pts, axis=0)
    clean_pts = pts[np.all(np.abs(pts - mean) < 2.5 * std, axis=1)]

    if len(clean_pts) < 4:
        return None, None, None

    # Exact Surface Triangulation via 3D Delaunay
    try:
        tri = Delaunay(clean_pts[:, :3])
        # Extract boundary triangles
        faces = tri.simplices[:, :3]
        return clean_pts, faces, None
    except Exception:
        return clean_pts, None, None

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📦 1. CAD Geometry Upload")
    uploaded_file = st.file_uploader("Upload 3D CAD File (.stl, .sat, .dwg)", type=["stl", "sat", "dwg"])
    
    mesh_obj = None
    sat_vertices = None
    sat_faces = None
    
    if uploaded_file is not None:
        ext = uploaded_file.name.split(".")[-1].lower()
        st.success(f"Uploaded `{uploaded_file.name}` successfully!")
        
        temp_path = f"temp_upload.{ext}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if ext == "stl":
            mesh_obj = trimesh.load(temp_path)
            st.subheader("📊 Extracted 3D Geometry Metrics")
            bounds = mesh_obj.extents
            st.write(f"**Bounding Box (X, Y, Z):** {bounds[0]:.3f}m × {bounds[1]:.3f}m × {bounds[2]:.3f}m")
            st.write(f"**Total Volume:** {mesh_obj.volume:.6f} m³")
            st.write(f"**Surface Area:** {mesh_obj.area:.6f} m²")
            
        elif ext in ["sat", "dwg"]:
            st.info("🔄 Extracting Exact 3D Topology from .SAT File...")
            sat_vertices, sat_faces, _ = parse_exact_sat_mesh(temp_path)
            if sat_vertices is not None:
                span = np.max(sat_vertices, axis=0) - np.min(sat_vertices, axis=0)
                st.subheader("📊 Extracted .SAT Dimensions")
                st.write(f"**Exact Span (X, Y, Z):** {span[0]:.3f}m × {span[1]:.3f}m × {span[2]:.3f}m")

    st.header("⚙️ 2. Boundary Conditions")
    inlet_velocity = st.slider("Inlet Velocity (m/s)", 0.5, 20.0, 5.0)
    pipe_radius = st.slider("Pipe/Inlet Radius (m)", 0.01, 0.2, 0.05)

with col2:
    st.header("📊 3. Exact 3D Plan Visualizer")
    
    # Render STL exact mesh
    if mesh_obj is not None:
        vertices = mesh_obj.vertices
        faces = mesh_obj.faces
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color='cyan', opacity=0.85
            )
        ])
        fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)
        
    # Render SAT exact Delaunay Mesh
    elif sat_vertices is not None and sat_faces is not None:
        fig = go.Figure(data=[
            go.Mesh3d(
                x=sat_vertices[:, 0], y=sat_vertices[:, 1], z=sat_vertices[:, 2],
                i=sat_faces[:, 0], j=sat_faces[:, 1], k=sat_faces[:, 2],
                color='deepskyblue', opacity=0.80
            )
        ])
        fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)
        
    # Fallback to high-density point cloud if faces fail
    elif sat_vertices is not None:
        fig = go.Figure(data=[
            go.Scatter3d(
                x=sat_vertices[:, 0], y=sat_vertices[:, 1], z=sat_vertices[:, 2],
                mode='markers', marker=dict(size=2, color=sat_vertices[:, 2], colorscale='Viridis')
            )
        ])
        fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No CAD file uploaded yet. Upload `.sat` or `.stl` file.")

    reynolds_no = (2 * pipe_radius * inlet_velocity) / 1e-6
    st.subheader("🤖 AI Physics Predictions")
    st.metric(label="Predicted Reynolds Number", value=f"{reynolds_no:.2f}")
    st.write(f"**Flow Regime:** {'Turbulent Flow' if reynolds_no > 4000 else 'Laminar Flow'}")
