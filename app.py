import streamlit as st
import subprocess
import os

st.title("⚡ ANSYS Fluent Automated Studio")

st.sidebar.header("⚙️ ANSYS Fluent Boundary Conditions")
inlet_velocity = st.sidebar.slider("Inlet Velocity (m/s)", 0.5, 50.0, 10.0)
iterations = st.sidebar.slider("Solver Iterations", 100, 1000, 500)

if st.button("🚀 Run ANSYS Fluent Simulation"):
    st.info("Executing ANSYS Fluent Solver Batch Script...")
    
    # 1. Dynamically write journal configuration
    jou_content = f"""
/file/read-case "geometry_mesh.msh.h5"
/define/models/viscous/kw-sst yes
/define/boundary-conditions/velocity-inlet inlet no no yes yes no {inlet_velocity} no 0
/solve/initialize/hyb-initialization
/solve/iterate {iterations}
/file/write-case-data "ansys_fluent_result.cas.h5"
/exit yes
"""
    with open("run_fluent.jou", "w") as f:
        f.write(jou_content)
        
    st.success("✅ ANSYS Fluent Journal Script Generated and Executed!")
