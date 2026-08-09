import streamlit as st
import numpy as np
import pandas as pd
import trimesh
import plotly.graph_objects as go
import re
import io
import uuid
import os

st.set_page_config(page_title="AI 3D CFD Studio", layout="wide")

st.title("⚡ AI-Powered 3D CAD CFD Analyzer")


def extract_sat_mesh(sat_bytes):
    """
    Approximate ACIS .sat point extraction + convex hull reconstruction.

    NOTE: This is NOT an exact B-rep reconstruction. ACIS .sat is a
    parametric NURBS-based format; a text-regex scan cannot recover
    exact surfaces. This function extracts numeric triples that look
    like coordinates, filters outliers, and wraps them in a convex
    hull for a rough visual preview only. Any concave features
    (bores, fillets, internal channels) will NOT appear in the mesh.
    """
    raw_coords = []
    text = sat_bytes.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        matches = re.findall(r'([-+]?\d*\.\d+|\d+)', line)
        if len(matches) >= 3:
            for i in range(0, len(matches) - 2, 3):
                try:
                    x, y, z = float(matches[i]), float(matches[i + 1]), float(matches[i + 2])
                    if abs(x) < 500 and abs(y) < 500 and abs(z) < 500:
                        raw_coords.append([x, y, z])
                except ValueError:
                    continue

    if len(raw_coords) < 10:
        return None, None

    pts = np.array(raw_coords)

    mean = np.mean(pts, axis=0)
    std = np.std(pts, axis=0)
    valid_mask = np.all(np.abs(pts - mean) < 2 * std, axis=1)
    clean_pts = pts[valid_mask]

    if len(clean_pts) < 10:
        return None, None

    try:
        cloud = trimesh.PointCloud(clean_pts)
        mesh = cloud.convex_hull
        return clean_pts, mesh
    except Exception:
        return clean_pts, None


# ---- session-scoped temp workspace ----
# Each browser session gets its own subfolder under /tmp so concurrent
# uploads (different tabs, different users on Streamlit Cloud) never
# read or overwrite each other's files.
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

SESSION_TMP_DIR = os.path.join("/tmp", "cfd_studio_uploads", st.session_state.session_id)
os.makedirs(SESSION_TMP_DIR, exist_ok=True)


col1, col2 = st.columns([1, 1])

with col1:
    st.header("📦 1. CAD Geometry Upload")
    uploaded_file = st.file_uploader("Upload 3D CAD File (.stl, .sat, .dwg)", type=["stl", "sat", "dwg"])

    mesh_obj = None
    sat_pts = None
    is_approximate = False

    if uploaded_file is not None:
        ext = uploaded_file.name.split(".")[-1].lower()
        st.success(f"Uploaded `{uploaded_file.name}` successfully!")

        file_bytes = uploaded_file.getvalue()

        if ext == "stl":
            # Load straight from the in-memory buffer — no shared disk
            # path, no risk of reading a different session's file.
            mesh_obj = trimesh.load(
                io.BytesIO(file_bytes),
                file_type="stl"
            )

        elif ext in ["sat", "dwg"]:
            st.info("🔄 Extracting approximate point cloud from .SAT geometry...")
            sat_pts, mesh_obj = extract_sat_mesh(file_bytes)
            is_approximate = True

        if mesh_obj is not None:
            st.subheader("📊 Extracted 3D Geometry Metrics")
            if is_approximate:
                st.warning(
                    "⚠️ This is a **convex hull approximation**, not an exact "
                    "reconstruction. Concave features (bores, fillets, internal "
                    "channels) are not represented. Metrics below describe the "
                    "convex hull, not the original solid."
                )
            bounds = mesh_obj.extents
            st.write(f"**Bounding Box (X, Y, Z):** {bounds[0]:.3f}m × {bounds[1]:.3f}m × {bounds[2]:.3f}m")
            st.write(f"**{'Approx. ' if is_approximate else ''}Volume:** {mesh_obj.volume:.6f} m³")
            st.write(f"**{'Approx. ' if is_approximate else ''}Surface Area:** {mesh_obj.area:.6f} m²")

    st.header("⚙️ 2. Boundary Conditions")
    inlet_velocity = st.slider("Inlet Velocity (m/s)", 0.5, 20.0, 5.0)
    pipe_radius = st.slider("Pipe/Inlet Radius (m)", 0.01, 0.2, 0.05)

with col2:
    st.header("📊 3. 3D Plan Visualizer")

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
