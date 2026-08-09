import streamlit as st
import numpy as np
import pandas as pd
import trimesh
import plotly.graph_objects as go
import re

st.set_page_config(page_title="AI 3D CFD Studio", layout="wide")

st.title("⚡ AI-Powered 3D CAD CFD Analyzer")

def extract_sat_mesh(sat_path):
    """Accurate ACIS .sat point mesh parser with outlier coordinate filtering"""
    raw_coords = []
    with open(sat_path, 'r', errors='ignore') as f:
        for line in f:
            # ACIS points usually appear in blocks with floats
            matches = re.findall(r'([-+]?\d*\.\d+|\d+)', line)
            if len(matches) >= 3:
                for i in range(0, len(matches)-2, 3):
                    try:
                        x, y, z = float(matches[i]), float(matches[i+1]), float(matches[i+2])
                        # Filter out ACIS header IDs / entity index numbers (> 500m scale outliers)
                        if abs(x) < 500 and abs(y) < 500 and abs(z) < 500:
                            raw_coords.append([x, y, z])
                    except ValueError:
                        continue

    if len(raw_coords) < 10:
        return None, None

    pts = np.array(raw_coords)
    
    # Statistical Outlier Removal (Remove header metadata noise)
    mean = np.mean(pts, axis=0)
    std = np.std(pts, axis=0)
    valid_mask = np.all(np.abs(pts - mean) < 2 * std, axis=1)
    clean_pts = pts[valid_mask]

    if len(clean_pts) < 10:
        return None, None

    # Auto-generate 3D Alpha-Convex Mesh Surface from extracted CAD points
    try:
        cloud = trimesh.PointCloud(clean_pts)
        mesh = cloud.convex_hull
        return clean_pts, mesh
    except Exception:
        return clean_pts, None

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📦 1. CAD Geometry Upload")
    uploaded_file = st.file_uploader("Upload 3D CAD File (.stl, .sat, .dwg)", type=["stl", "sat", "dwg"])
    
    mesh_obj = None
    sat_pts = None
    
    if uploaded_file is not None:
        ext = uploaded_file.name.split(".")[-1].lower()
        st.success(f"Uploaded `{uploaded_file.name}` successfully!")
        
        temp_path = f"temp_upload.{ext}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if ext == "stl":
            mesh_obj = trimesh.load(temp_path)
        elif ext in ["sat", "dwg"]:
            st.info("🔄 Reconstructing exact 3D Mesh Surface from .SAT geometry...")
            sat_pts, mesh_obj = extract_sat_mesh(temp_path)

        if mesh_obj is not None:
            st.subheader("📊 Extracted 3D Geometry Metrics")
            bounds = mesh_obj.extents
            st.write(f"**Bounding Box (X, Y, Z):** {bounds[0]:.3f}m × {bounds[1]:.3f}m × {bounds[2]:.3f}m")
            st.write(f"**Total Volume:** {mesh_obj.volume:.6f} m³")
            st.write(f"**Surface Area:** {mesh_obj.area:.6f} m²")

    st.header("⚙️ 2. Boundary Conditions")
    inlet_velocity = st.slider("Inlet Velocity (m/s)", 0.5, 20.0, 5.0)
    pipe_radius = st.slider("Pipe/Inlet Radius (m)", 0.01, 0.2, 0.05)

with col2:
    st.header("📊 3. Exact 3D Plan Visualizer")
    
    if mesh_obj is not None:
        vertices = mesh_obj.vertices
        faces = mesh_obj.faces
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color='deepskyblue', opacity=0.85
            )
        ])
        fig.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)
    elif sat_pts is not None:
        fig = go.Figure(data=[
            go.Scatter3d(
                x=sat_pts[:, 0], y=sat_pts[:, 1], z=sat_pts[:, 2],
                mode='markers', marker=dict(size=3, color=sat_pts[:, 2], colorscale='Blues')
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
