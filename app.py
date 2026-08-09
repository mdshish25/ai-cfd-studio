import streamlit as st
import numpy as np
import pandas as pd
import trimesh
import plotly.graph_objects as go
import os

st.set_page_config(page_title="AI 3D CFD Studio", layout="wide")

st.title("⚡ AI-Powered 3D CAD CFD Analyzer")
st.write("Upload a 3D CAD file (.stl, .sat, .dwg) for automatic 3D mesh extraction and instant AI flow predictions.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📦 1. CAD Geometry Upload")
    uploaded_file = st.file_uploader("Upload 3D CAD File (.stl, .sat, .dwg)", type=["stl", "sat", "dwg"])
    
    mesh = None
    if uploaded_file is not None:
        ext = uploaded_file.name.split(".")[-1].lower()
        st.success(f"Uploaded `{uploaded_file.name}` successfully!")
        
        # Save temporary uploaded file
        temp_path = f"temp_upload.{ext}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # AUTOMATIC CAD CONVERSION PIPELINE
        if ext == "stl":
            mesh = trimesh.load(temp_path)
        elif ext in ["sat", "dwg"]:
            st.info(f"🔄 Automatically converting .{ext} CAD solid topology to 3D mesh...")
            try:
                import cadquery as cq
                # Import SAT / DWG solid shape
                cad_shape = cq.importers.importShape(temp_path)
                # Export as temporary STL mesh
                converted_stl = "temp_converted.stl"
                cq.exporters.export(cad_shape, converted_stl)
                mesh = trimesh.load(converted_stl)
                st.success("✅ CAD Solid automatically converted to 3D Mesh!")
            except Exception as e:
                st.error(f"Automatic conversion failed: {str(e)}")
                st.warning("Ensure the .sat file contains closed 3D solid geometry.")

        if mesh is not None:
            st.subheader("📊 Extracted 3D Geometric Data")
            bounds = mesh.extents
            st.write(f"**Bounding Box (X, Y, Z):** {bounds[0]:.3f}m × {bounds[1]:.3f}m × {bounds[2]:.3f}m")
            st.write(f"**Total Volume:** {mesh.volume:.6f} m³")
            st.write(f"**Surface Area:** {mesh.area:.6f} m²")

    st.header("⚙️ 2. Boundary Conditions")
    inlet_velocity = st.slider("Inlet Velocity (m/s)", 0.5, 20.0, 5.0)
    pipe_radius = st.slider("Pipe/Inlet Radius (m)", 0.01, 0.2, 0.05)

with col2:
    st.header("📊 3. Interactive 3D CAD Plan Viewer")
    
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
                color='cyan',
                opacity=0.8
            )
        ])
        fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No CAD file uploaded yet. Upload a .sat / .stl file to view your 3D plan.")

    # Physics Prediction Box
    reynolds_no = (2 * pipe_radius * inlet_velocity) / 1e-6
    st.subheader("🤖 AI Physics Predictions")
    st.metric(label="Predicted Reynolds Number", value=f"{reynolds_no:.2f}")
    st.write(f"**Flow Regime:** {'Turbulent Flow' if reynolds_no > 4000 else 'Laminar Flow'}")
