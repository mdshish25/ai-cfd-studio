import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import pickle
import os

st.set_page_config(page_title="AI CFD 3D Studio - Phase 1", layout="wide")

# Title & Header
st.title("⚡ AI-Powered 3D CFD Simulation Tool")
st.write("Upload a 3D CAD file (.STL) or adjust parameters to see instant AI predictions.")

# Layout: 2 Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📦 1. CAD Geometry (.STL Upload)")
    uploaded_file = st.file_uploader("Upload 3D STL File", type=["stl"])
    
    mesh = None
    if uploaded_file is not None:
        with open("temp_cad.stl", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            mesh = trimesh.load("temp_cad.stl")
            st.success(f"CAD File Loaded Successfully!")
            st.write(f"**Vertices:** {len(mesh.vertices)} | **Faces:** {len(mesh.faces)}")
            
            # 3D Geometry Preview
            fig_3d, ax_3d = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(5, 4))
            ax_3d.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], 
                               triangles=mesh.faces, cmap='Spectral', edgecolor='none')
            ax_3d.set_title("Loaded 3D Geometry Preview")
            ax_3d.axis('off')
            st.pyplot(fig_3d)
        except Exception as e:
            st.error(f"Error loading STL file: {e}")
    else:
        st.info("No CAD file uploaded. Showing default domain demo.")

    st.subheader("⚙️ 2. Boundary Conditions & Display Options")
    inlet_velocity = st.slider("Inlet Velocity (m/s)", 0.5, 10.0, 2.0, 0.5)
    pipe_radius = st.slider("Pipe/Inlet Radius (m)", 0.01, 0.2, 0.05, 0.01)
    length = st.slider("Domain Length (m)", 0.1, 2.0, 0.5, 0.1)

    # NEW: Plot Customization Options
    plot_type = st.radio("Select Physics Field Display:", ["Velocity Field", "Pressure Drop Field"])
    show_streamlines = st.checkbox("Overlay Flow Streamlines", value=True)

with col2:
    st.subheader("📊 3. AI CFD Prediction Results")
    
    if os.path.exists("cfd_ai_model.pkl"):
        with open("cfd_ai_model.pkl", "rb") as f:
            ai_model = pickle.load(f)
        
        # AI Inference
        input_data = np.array([[inlet_velocity, pipe_radius]])
        prediction = ai_model.predict(input_data)
        
        predicted_reynolds = prediction[0][0]
        predicted_peak_vel = prediction[0][1]

        # Display Key Metrics
        m1, m2 = st.columns(2)
        m1.metric("AI Predicted Reynolds Number", f"{predicted_reynolds:.2f}")
        
        # Approximate Pressure Drop using Bernoulli/Poiseuille equation
        delta_p = 8 * 0.001 * length * predicted_peak_vel / (pipe_radius**2)
        m2.metric("Est. Pressure Drop (Pa)", f"{delta_p:.2f}")

        if predicted_reynolds < 2300:
            st.success("Regime: Laminar Flow")
        else:
            st.warning("Regime: Turbulent Flow")

        # 2D Grid Construction
        x = np.linspace(0, length, 100)
        r = np.linspace(-pipe_radius, pipe_radius, 50)
        X, R = np.meshgrid(x, r)

        # Calculate Velocity & Pressure Fields
        u_velocity = predicted_peak_vel * (1 - (R / pipe_radius)**2)
        v_velocity = np.zeros_like(u_velocity)  # 1D axially developed flow
        pressure_field = delta_p * (1 - X / length)

        fig, ax = plt.subplots(figsize=(8, 4))

        if plot_type == "Velocity Field":
            heatmap = ax.contourf(X, R, u_velocity, levels=50, cmap='jet')
            fig.colorbar(heatmap, ax=ax, label='Velocity (m/s)')
            ax.set_title("Velocity Distribution Field")
        else:
            heatmap = ax.contourf(X, R, pressure_field, levels=50, cmap='coolwarm')
            fig.colorbar(heatmap, ax=ax, label='Pressure (Pa)')
            ax.set_title("Pressure Distribution Field")

        # Overlay Streamlines if enabled
        if show_streamlines:
            ax.streamplot(X, R, u_velocity, v_velocity, color='white', linewidth=0.8, density=0.8)

        ax.set_xlabel("Length (m)")
        ax.set_ylabel("Radius/Height (m)")
        st.pyplot(fig)

        # NEW: Download Report Section
        st.subheader("💾 4. Export Simulation Data")
        report_text = f"""--- AI CFD SIMULATION REPORT ---
Inlet Velocity: {inlet_velocity} m/s
Pipe Radius: {pipe_radius} m
Domain Length: {length} m
--------------------------------
Predicted Reynolds Number: {predicted_reynolds:.2f}
Estimated Pressure Drop: {delta_p:.2f} Pa
Flow Regime: {'Laminar' if predicted_reynolds < 2300 else 'Turbulent'}
"""
        st.download_button(
            label="📄 Download Summary Report (.txt)",
            data=report_text,
            file_name="cfd_simulation_summary.txt",
            mime="text/plain"
        )
    else:
        st.error("AI Model 'cfd_ai_model.pkl' not found!")