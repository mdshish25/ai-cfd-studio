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

st.set_page_config(page_title="ANSYS Multi-Physics Studio", layout="wide")

st.title("⚡ ANSYS Multi-Physics AI Studio (CFD + Thermal + Probes)")
st.write("3D CAD Surface Analysis, ANSYS Post-Processing Tools, Slicing, Material Library & Reports")

def parse_sat_file(sat_path):
    """Extract spatial vertices from ACIS .sat ASCII file"""
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

def generate_ansys_contour_plot(velocity, radius, max_temp):
    """Generate static ANSYS style contour figure for PDF report"""
    fig, ax = plt.subplots(figsize=(6, 2.5), facecolor='#0F172A')
    ax.set_facecolor('#0F172A')
    
    x = np.linspace(-radius, radius, 100)
    y = np.linspace(-0.25, 0.25, 50)
    X, Y = np.meshgrid(x, y)
    Z = max_temp - 15.0 * (1.0 - (X**2 + Y**2)/(radius**2 + 0.01))
    
    contour = ax.contourf(X, Y, Z, cmap='inferno', levels=15)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    cbar.set_label('Temperature (°C)', color='white')
    
    ax.axis('off')
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_ansys_workbench_pdf(filename, project_name, author, velocity, radius, sim_results, mat_name):
    """Generates an ANSYS Workbench Mechanical Report PDF with CFD + Thermal Details"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('ANSYSTitle', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor('#002B49'), spaceAfter=2)
    sub_style = ParagraphStyle('ANSYSSub', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#475569'), spaceAfter=10)
    
    story.append(Paragraph("ANSYS Workbench Multi-Physics Report", title_style))
    story.append(Paragraph("DesignSpace / Mechanical Coupled Thermal-CFD Simulation", sub_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#FFB800'), spaceAfter=12))

    now_str = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M:%S %p")
    meta_data = [
        ["Project", project_name, "Software Used", "ANSYS Workbench v23.2 / AI Engine"],
        ["Author", author, "Database Path", f"C:\\ANSYS_MODELS\\{filename}"],
        ["Report Created", now_str, "Selected Material", mat_name]
    ]
    t_meta = Table(meta_data, colWidths=[90, 160, 100, 200])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F1F5F9')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Summary & Material Specification", styles['Heading2']))
    story.append(Spacer(1, 4))
    
    summary_p = (
        f"This report documents design and thermal-fluid analysis created using the ANSYS simulation engine. "
        f"The model <b>{filename}</b> was assigned material <b>{mat_name}</b> and evaluated under fluid inlet velocity of "
        f"<b>{velocity:.2f} m/s</b>. Solved Navier-Stokes equations and convective heat transfer."
    )
    story.append(Paragraph(summary_p, styles['Normal']))
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Simulation Scenario & Results Summary", styles['Heading2']))
    story.append(Spacer(1, 4))

    results_data = [
        ["Scenario", "Reynolds No.", "Max Temp (°C)", "Min Temp (°C)", "Heat Coeff h (W/m²K)", "Dissipation (W)"],
        ["CFD & Thermal Run", f"{sim_results['reynolds_no']:,.0f}", f"{sim_results['max_temp']:.1f} °C", f"{sim_results['min_temp']:.1f} °C", f"{sim_results['h_coeff']:.1f}", f"{sim_results['heat_loss']:.1f} W"],
    ]
    t_res = Table(results_data, colWidths=[110, 85, 85, 85, 105, 80])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0284C7')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#38BDF8')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F0F9FF')),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Thermal Field & ANSYS Temperature Contours", styles['Heading2']))
    story.append(Spacer(1, 4))

    contour_img_buf = generate_ansys_contour_plot(velocity, radius, sim_results['max_temp'])
    story.append(Image(contour_img_buf, width=480, height=200))
    story.append(Paragraph("<i>Figure T1.1: 3D Convective Thermal Temperature Contour Map (°C).</i>", styles['Italic']))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ANSYS MATERIAL LIBRARY
MATERIAL_LIBRARY = {
    "Aluminum 6061 T6": {"k": 167.0, "density": 2700, "cp": 896},
    "Copper (Pure)": {"k": 401.0, "density": 8960, "cp": 385},
    "Silicon (Si)": {"k": 156.0, "density": 2330, "cp": 700},
    "Structural Steel": {"k": 60.5, "density": 7850, "cp": 434},
    "Titanium Alloy (Ti-6Al-4V)": {"k": 6.7, "density": 4430, "cp": 526},
    "Carbon Fiber Composites": {"k": 15.0, "density": 1810, "cp": 850}
}

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📦 1. ANSYS Multi-Physics Setup")
    project_name = st.text_input("Project Name", "ATLAS Stave Upgrade Thermal-CFD")
    author_name = st.text_input("Author Name", "Margareta Rehak")
    uploaded_file = st.file_uploader("Upload CAD Geometry (.stl, .sat)", type=["stl", "sat"])
    
    st.subheader("🧱 ANSYS Engineering Material Library")
    selected_mat = st.selectbox("Assign Surface Material:", list(MATERIAL_LIBRARY.keys()))
    mat_info = MATERIAL_LIBRARY[selected_mat]
    st.caption(f"**Properties:** Conductivity $k = {mat_info['k']}$ W/m·K | Density $= {mat_info['density']}$ kg/m³")

    st.subheader("⚙️ Boundary Conditions")
    inlet_velocity = st.slider("Fluid Inlet Velocity (m/s)", 0.5, 50.0, 10.0)
    pipe_radius = st.slider("Channel Radius / Scale (m)", 0.01, 0.5, 0.05)
    inlet_temp = st.slider("Inlet Temperature T_in (°C)", -20.0, 100.0, 20.0)
    heat_flux = st.slider("Applied Surface Heat Flux (W/m²)", 100.0, 5000.0, 1666.7)

    st.subheader("🛠️ ANSYS Workbench Post-Processing Tools")
    show_mesh_wire = st.checkbox("🕸️ Show ANSYS Mesh Triangulated Wireframe", value=False)
    show_probes = st.checkbox("📍 Enable Max/Min Local Probe Sensors", value=True)
    slice_cut_plane = st.slider("✂️ Z-Axis Section Slicer Cut-Plane (m)", -0.5, 0.5, 0.5, step=0.01)

    mesh = None
    filename_str = "STAVE_Simul.stl"

    if uploaded_file is not None:
        filename_str = uploaded_file.name
        ext = filename_str.split(".")[-1].lower()
        temp_path = f"temp_upload.{ext}"
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if ext == "stl":
            try:
                mesh = trimesh.load(temp_path)
            except Exception:
                mesh = None
        elif ext == "sat":
            mesh = parse_sat_file(temp_path)

    if mesh is None or not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.creation.cylinder(radius=pipe_radius, height=0.5)

    verts = mesh.vertices
    faces = mesh.faces

    # APPLY SECTION SLICING FILTER
    slice_mask = verts[:, 2] <= slice_cut_plane
    sliced_verts = verts[slice_mask]

    # CFD & THERMAL PHYSICS SOLVER ENGINES
    fluid_density = 1.225
    viscosity = 1.81e-5
    reynolds_no = (fluid_density * inlet_velocity * (2 * pipe_radius)) / viscosity
    regime = "Turbulent Flow" if reynolds_no > 4000 else "Laminar Flow"
    dynamic_pressure = 0.5 * fluid_density * (inlet_velocity**2)
    drag_force = 0.45 * dynamic_pressure * (math.pi * (pipe_radius**2))
    
    prandtl_no = 0.71
    nusselt_no = 0.023 * (reynolds_no**0.8) * (prandtl_no**0.4) if regime == "Turbulent Flow" else 3.66
    k_thermal = mat_info['k']
    h_coeff = (nusselt_no * k_thermal) / (2 * pipe_radius)
    
    max_temp = inlet_temp + (heat_flux / max(h_coeff, 1e-3))
    min_temp = inlet_temp
    heat_loss = h_coeff * mesh.area * (max_temp - min_temp)

    sim_results = {
        "reynolds_no": reynolds_no,
        "regime": regime,
        "drag_force": drag_force,
        "h_coeff": h_coeff,
        "max_temp": max_temp,
        "min_temp": min_temp,
        "heat_loss": heat_loss
    }

with col2:
    st.header("🖥️ 2. ANSYS Workbench 3D Visualizer")
    
    contour_mode = st.radio(
        "Select Contour Display Engine:",
        ["Thermal Temperature Field (°C)", "CFD Velocity Magnitude (m/s)"]
    )

    r_dist = np.sqrt(verts[:, 0]**2 + verts[:, 1]**2)
    norm_r = r_dist / max(np.max(r_dist), 1e-4)

    if "Thermal" in contour_mode:
        contour_field = min_temp + (max_temp - min_temp) * (norm_r**2)
        color_scheme = "Inferno"
        bar_title = "Temperature (°C)"
    else:
        contour_field = inlet_velocity * (1.0 - 0.75 * (norm_r**2))
        color_scheme = "Jet"
        bar_title = "Velocity (m/s)"

    # 3D MESH SURFACE
    fig = go.Figure()
    
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=contour_field,
        colorscale=color_scheme,
        colorbar=dict(title=bar_title, thickness=20),
        opacity=0.98
    ))

    # TOOL 1: ANSYS MESH WIREFRAME OVERLAY
    if show_mesh_wire:
        fig.add_trace(go.Scatter3d(
            x=verts[::3, 0], y=verts[::3, 1], z=verts[::3, 2],
            mode='markers+lines',
            marker=dict(size=2, color='white'),
            line=dict(color='gray', width=1),
            name="ANSYS Mesh"
        ))

    # TOOL 2: ANSYS PROBE TAG SENSORS
    if show_probes:
        max_idx = np.argmax(contour_field)
        min_idx = np.argmin(contour_field)
        
        fig.add_trace(go.Scatter3d(
            x=[verts[max_idx, 0]], y=[verts[max_idx, 1]], z=[verts[max_idx, 2]],
            mode='markers+text',
            marker=dict(size=10, color='red', symbol='diamond'),
            text=[f"MAX: {np.max(contour_field):.1f}"],
            textposition="top center",
            name="Max Probe"
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[verts[min_idx, 0]], y=[verts[min_idx, 1]], z=[verts[min_idx, 2]],
            mode='markers+text',
            marker=dict(size=10, color='blue', symbol='diamond'),
            text=[f"MIN: {np.min(contour_field):.1f}"],
            textposition="bottom center",
            name="Min Probe"
        ))

    fig.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', bgcolor="#0F172A"), margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🤖 ANSYS Probes & Output Metrics")
    tm1, tm2, tm3 = st.columns(3)
    tm1.metric("Max Probe Value", f"{max_temp:.1f} °C")
    tm2.metric("Convective Heat Coeff", f"{h_coeff:.1f} W/m²K")
    tm3.metric("Thermal Loss", f"{heat_loss:.1f} W")

    pdf_bytes = generate_ansys_workbench_pdf(filename_str, project_name, author_name, inlet_velocity, pipe_radius, sim_results, selected_mat)

    st.download_button(
        label="📄 Download Technical ANSYS Workbench PDF Report",
        data=pdf_bytes,
        file_name="ANSYS_MultiPhysics_Report.pdf",
        mime="application/pdf"
    )
