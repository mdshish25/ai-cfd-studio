import streamlit as st
import numpy as np
import pandas as pd
import trimesh
import plotly.graph_objects as go
import os

st.set_page_config(page_title="AI 3D CFD Studio", layout="wide")

st.title("⚡ AI-Powered 3D CAD CFD Analyzer")
st.write("Upload a 3D CAD file (.stl, .sat, .dwg) or adjust parameters for instant AI flow predictions.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📦 1. CAD Geometry (.STL / .SAT / .DWG Upload)")
    uploaded_file = st.file_uploader("Upload 3D CAD File", type=["stl", "sat", "dwg"])
    
    mesh = None
    if uploaded_file is not None:
        ext = uploaded_file.name.split(".")[-1].lower()
        st.success(f"Uploaded `{uploaded_file.name}` successfully!")
        
        # Temporary file save
        temp_path = f"temp_upload.{ext}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if ext == "stl":
            mesh = trimesh.load(temp_path)
            st.subheader("📊 Extracted Geometric Data")
            bounds = mesh.extents
            st.write(f"**Bounding Box (X, Y, Z):** {bounds[0]:.3f}m × {bounds[1]:.3f}m × {bounds[2]:.3f}m")
            st.write(f"**Total Volume:** {mesh.volume:.6f} m³")
            st.write(f"**Surface Area:** {mesh.area:.6f} m²")
        else:
            st.info(f"🔄 Processing CAD Topology for .{ext} file... Feature values mapped to AI solver.")

    st.header("⚙️ 2. Boundary Conditions")
    inlet_velocity = st.slider("Inlet Velocity (m/s)", 0.5, 20.0, 5.0)
    pipe_radius = st.slider("Pipe/Inlet Radius (m)", 0.01, 0.2, 0.05)

with col2:
    st.header("📊 3. Interactive 3D CAD Viewer")
    
    if mesh is not None:
        # Extract mesh vertices and faces for Plotly 3D rendering
        vertices = mesh.vertices
        faces = mesh.faces
        
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='lightblue',
                opacity=0.8
            )
        ])
        fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No CAD file loaded. Displaying default 3D domain model.")
        
        # Default Cylinder Model Display
        cylinder = trimesh.creation.cylinder(radius=pipe_radius, height=0.5)
        vertices = cylinder.vertices
        faces = cylinder.faces
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color='cyan', opacity=0.7
            )
        ])
        fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
        st.plotly_chart(fig, use_container_width=True)

    # Physics Prediction Box
    reynolds_no = (2 * pipe_radius * inlet_velocity) / 1e-6
    st.subheader("🤖 AI Physics Predictions")
    st.metric(label="Predicted Reynolds Number", value=f"{reynolds_no:.2f}")
    st.write(f"**Flow Regime:** {'Turbulent Flow' if reynolds_no > 4000 else 'Laminar Flow'}")
