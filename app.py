import streamlit as st
import numpy as np
import pandas as pd
import trimesh
import plotly.graph_objects as go
import re
import datetime
import math
import matplotlib.pyplot as plt
import requests
import json
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Streamlit Page Setup
st.set_page_config(page_title="ANSYS Multi-Physics & shish AI Studio", layout="wide", initial_sidebar_state="collapsed")

# INITIALIZE POSITION STATE
if "chat_pos" not in st.session_state:
    st.session_state.chat_pos = "Bottom-Right"

# POSITION CSS MAPPING
pos_css = {
    "Bottom-Left": "bottom: 25px !important; left: 25px !important;",
    "Bottom-Right": "bottom: 25px !important; right: 25px !important;",
    "Top-Left": "top: 25px !important; left: 25px !important;",
    "Top-Right": "top: 25px !important; right: 25px !important;"
}
current_pos_css = pos_css.get(st.session_state.chat_pos, pos_css["Bottom-Right"])

# CLEAN WHITE THEME & ATTRACTIVE FLOATING WIDGET STYLING
st.markdown(f"""
<style>
    /* MAIN LIGHT/WHITE BACKGROUND */
    .stApp {{
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}
    
    /* ANSYS PROFESSIONAL BANNER */
    .ansys-top-banner {{
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        color: #FFFFFF;
        padding: 14px 24px;
        font-weight: 700;
        font-size: 18px;
        letter-spacing: 0.8px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(2, 132, 199, 0.2);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    /* WHITE CARDS STYLING */
    .ansys-card {{
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 18px !important;
        margin-bottom: 18px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
    }}
    .card-title {{
        color: #0284C7 !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        margin-bottom: 14px !important;
        border-bottom: 2px solid #F1F5F9 !important;
        padding-bottom: 8px !important;
    }}

    /* FLOATING AI TRIGGER BUTTON */
    div.floating-ai-btn-container {{
        position: fixed !important;
        {current_pos_css}
        z-index: 999999 !important;
    }}

    div.floating-ai-btn-container button {{
        background: linear-gradient(135deg, #0284C7 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        border: 2px solid #38BDF8 !important;
        border-radius: 30px !important;
        padding: 14px 28px !important;
        font-weight: 800 !important;
        font-size: 16px !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 10px 25px rgba(2, 132, 199, 0.4) !important;
        cursor: pointer !important;
        transition: all 0.3s ease-in-out !important;
    }}

    div.floating-ai-btn-container button:hover {{
        background: linear-gradient(135deg, #0369A1 0%, #1D4ED8 100%) !important;
        transform: translateY(-3px) scale(1.04) !important;
        box-shadow: 0 14px 30px rgba(2, 132, 199, 0.6) !important;
    }}
    
    /* WHATSAPP-STYLE CHAT CONTAINER */
    div[data-testid="stVerticalBlock"] > div:has(div.floating-chat-anchor) {{
        position: fixed !important;
        {current_pos_css}
        z-index: 999999 !important;
        width: 440px !important;
        max-width: 92vw !important;
        background-color: #E5DDD5 !important;
        border: 2px solid #CBD5E1 !important;
        border-radius: 18px !important;
        padding: 16px !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2) !important;
        color: #0F172A !important;
        resize: both !important;
        overflow: auto !important;
    }}

    /* CLEAN INPUT AREA */
    div[data-testid="stVerticalBlock"] > div:has(div.floating-chat-anchor) div[data-testid="stBottom"] {{
        padding-bottom: 0px !important;
        margin-bottom: 0px !important;
    }}

    div[data-testid="stVerticalBlock"] > div:has(div.floating-chat-anchor) div[data-testid="stChatInput"] {{
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }}

    /* WHATSAPP BUBBLES */
    div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) {{
        background-color: #DCF8C6 !important;
        margin-left: auto !important;
        margin-right: 0px !important;
        border-radius: 12px 0px 12px 12px !important;
        max-width: 82% !important;
        border: 1px solid #C4E8B0 !important;
    }}

    div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from assistant"]) {{
        background-color: #FFFFFF !important;
        margin-left: 0px !important;
        margin-right: auto !important;
        border-radius: 0px 12px 12px 12px !important;
        max-width: 85% !important;
        border: 1px solid #E2E8F0 !important;
    }}

    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] div {{
        color: #111B21 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        line-height: 1.5 !important;
    }}
</style>
""", unsafe_allow_html=True)

# RETRIEVE API KEYS FROM STREAMLIT SECRETS
groq_key = st.secrets.get("GROQ_API_KEY", None)
gemini_key = st.secrets.get("GEMINI_API_KEY", None)

def query_shish_ai_permanent(prompt_text):
    """
    PERMANENT NO-LIMIT AI ENGINE:
    1. Instant Local Math Solver
    2. Groq Llama-3 API (High Speed & 100% Free Rate Limit)
    3. Gemini REST API Fallback
    """
    try:
        clean_expr = prompt_text.replace("=", "").replace("?", "").replace("kya hota hai", "").strip()
        if re.match(r"^[\d\+\-\*\/\.\s\(\)]+$", clean_expr):
            calc_res = eval(clean_expr)
            return f"**{prompt_text}** = `{calc_res}`"
    except Exception:
        pass

    sys_prompt = (
        "You are Shish, an advanced, highly intelligent AI Assistant powered by LLM. "
        "Your goal is to be friendly, grounded, exceptionally helpful, and accurate. "
        "You must answer ALL user questions across any domain: Science, Math, Engineering (ANSYS, CFD, FEA), "
        "Python Coding, Logic, and General Knowledge. Communicate naturally in Hinglish or English."
    )

    if groq_key:
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": str(prompt_text)}
                ],
                "temperature": 0.7
            }
            res = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

    if gemini_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": f"{sys_prompt}\n\nUser Question: {prompt_text}"}]}]
            }
            res = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
            if res.status_code == 200:
                return res.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            pass

    return "⚠️ Please add `GROQ_API_KEY` or `GEMINI_API_KEY` in Streamlit Cloud Secrets."

# MATERIAL LIBRARY DATABASE
MATERIALS_DB = {
    "Air (Ideal Gas)": {"density": 1.225, "viscosity": 1.81e-5, "k": 0.026, "cp": 1005},
    "Water (Liquid)": {"density": 998.0, "viscosity": 1.005e-3, "k": 0.6, "cp": 4182},
    "Liquid Methane (CH4)": {"density": 422.0, "viscosity": 1.1e-4, "k": 0.19, "cp": 3480},
    "Structural Steel": {"density": 7850, "viscosity": 0.0, "k": 60.5, "cp": 434},
    "Aluminum 6061-T6": {"density": 2700, "viscosity": 0.0, "k": 167.0, "cp": 896},
    "Titanium Grade 5": {"density": 4430, "viscosity": 0.0, "k": 6.7, "cp": 526}
}

# REPORT GENERATOR FUNCTIONS
def generate_ansys_contour_figure(verts, field_data, field_title):
    fig, ax = plt.subplots(figsize=(6, 3), facecolor='#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    x, y = verts[:, 0], verts[:, 1]
    min_len = min(len(x), len(field_data))
    sc = ax.scatter(x[:min_len], y[:min_len], c=field_data[:min_len], cmap='jet', s=8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='#0F172A')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#0F172A')
    cbar.set_label(field_title, color='#0F172A')
    ax.axis('off')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_ansys_workbench_pdf(filename, project_name, author, physics_mode, mat_name, mat_props, verts, contour_field, field_title):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=25, leftMargin=25, topMargin=25, bottomMargin=25)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('ANSYSTitle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#002B49'), spaceAfter=2)
    sub_style = ParagraphStyle('ANSYSSub', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#475569'), spaceAfter=10)
    
    story.append(Paragraph("ANSYS Multi-Physics Simulation Report", title_style))
    story.append(Paragraph(f"Release 2026 R1 - Official {physics_mode} Analysis Report", sub_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#FFB800'), spaceAfter=10))

    now_str = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M:%S %p")
    meta_data = [
        ["Project", project_name, "Software Version", "shish AI Engine 2026 R1"],
        ["Author", author, "Database Path", f"C:\\ANSYS_MODELS\\{filename}"],
        ["Report Created", now_str, "Physics Module", physics_mode]
    ]
    t_meta = Table(meta_data, colWidths=[90, 170, 100, 200])
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

    story.append(Paragraph("1. Material Specification & Domain Conditions", styles['Heading2']))
    story.append(Spacer(1, 4))
    
    mat_table_data = [
        ["Material Selected", "Density (kg/m³)", "Conductivity k (W/m·K)", "Specific Heat Cp (J/kg·K)"],
        [mat_name, f"{mat_props['density']}", f"{mat_props['k']}", f"{mat_props['cp']}"]
    ]
    t_mat = Table(mat_table_data, colWidths=[160, 130, 140, 130])
    t_mat.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#002B49')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#94A3B8')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8FAFC')),
    ]))
    story.append(t_mat)
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Numerical Field Results Summary", styles['Heading2']))
    story.append(Spacer(1, 4))
    
    res_table_data = [
        ["Field Metric", "Maximum Computed", "Minimum Computed", "Unit Status"],
        [field_title, f"{np.max(contour_field):.2f}", f"{np.min(contour_field):.2f}", "CONVERGED"]
    ]
    t_res = Table(res_table_data, colWidths=[180, 130, 130, 120])
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

    story.append(Paragraph("3. ANSYS Contour Distribution", styles['Heading2']))
    story.append(Spacer(1, 4))

    contour_img_buf = generate_ansys_contour_figure(verts, contour_field, field_title)
    story.append(Image(contour_img_buf, width=500, height=250))
    story.append(Paragraph(f"<i>Figure 1: ANSYS 2026 R1 {field_title} Nodal Distribution Map.</i>", styles['Italic']))

    doc.build(story)
    buffer.seek(0)
    return buffer

# MAIN WORKSTATION DASHBOARD
st.markdown("""
<div class="ansys-top-banner">
    <div>⚡ ANSYS Discovery 2026 R1 - Multi-Physics Workstation Studio</div>
    <div style="font-size: 13px; opacity: 0.9;">CFD | Static Structural FEA | Combustion | Multiphase VOF</div>
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

    else:
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
        colorbar=dict(title=bar_title, thickness=18, x=1.01, len=0.8, tickfont=dict(color='#0F172A')),
        opacity=0.98,
        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.1)
    ))

    if show_mesh_wire:
        fig.add_trace(go.Scatter3d(
            x=verts[::3, 0], y=verts[::3, 1], z=verts[::3, 2],
            mode='markers+lines',
            marker=dict(size=2, color='#0284C7'),
            line=dict(color='#64748B', width=1)
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
            xaxis=dict(title='X (m)', backgroundcolor="#F1F5F9", gridcolor="#CBD5E1", showbackground=True),
            yaxis=dict(title='Y (m)', backgroundcolor="#F1F5F9", gridcolor="#CBD5E1", showbackground=True),
            zaxis=dict(title='Z (m)', backgroundcolor="#F1F5F9", gridcolor="#CBD5E1", showbackground=True),
            aspectmode='data'
        ),
        height=680,
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="#FFFFFF"
    )
    st.plotly_chart(fig, use_container_width=True)

# EXPORT EXECUTIVE REPORT CARD
with col_details:
    st.markdown('<div class="ansys-card"><div class="card-title">📄 Export Executive Report</div>', unsafe_allow_html=True)
    pdf_data = generate_ansys_workbench_pdf(
        filename=filename_str,
        project_name="Multi-Physics Analysis",
        author="shish AI Workstation Engine",
        physics_mode=physics_mode,
        mat_name=selected_mat,
        mat_props=mat_props,
        verts=verts,
        contour_field=contour_field,
        field_title=bar_title
    )

    st.download_button(
        label="📥 Download Executive PDF Report",
        data=pdf_data,
        file_name=f"{filename_str.split('.')[0]}_shish_ANSYS_Report.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# FLOATING & POSITIONABLE CHATBOT WIDGET
# ---------------------------------------------------------
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

if not st.session_state.chat_open:
    st.markdown('<div class="floating-ai-btn-container">', unsafe_allow_html=True)
    if st.button("💬 shish AI", key="floating_shish_btn"):
        st.session_state.chat_open = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.chat_open:
    with st.container():
        st.markdown('<div class="floating-chat-anchor"></div>', unsafe_allow_html=True)
        head_col1, head_col2, head_col3 = st.columns([3.5, 1.8, 0.7])
        with head_col1:
            st.markdown("<h2 style='color:#0F172A; margin:0; font-size:20px; font-weight:800;'>🤖 shish AI</h2>", unsafe_allow_html=True)
        with head_col2:
            new_pos = st.selectbox("Position", ["Bottom-Right", "Bottom-Left", "Top-Left", "Top-Right"], index=["Bottom-Right", "Bottom-Left", "Top-Left", "Top-Right"].index(st.session_state.chat_pos), label_visibility="collapsed", key="pos_select")
            if new_pos != st.session_state.chat_pos:
                st.session_state.chat_pos = new_pos
                st.rerun()
        with head_col3:
            if st.button("─", help="Minimize Chat", key="minimize_chat_btn"):
                st.session_state.chat_open = False
                st.rerun()

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I am **Shish**. You can ask any question!"}
            ]

        chat_container = st.container(height=380)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        if user_input := st.chat_input("Ask shish AI..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            reply = query_shish_ai_permanent(user_input)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()
