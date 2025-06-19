import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Streamlit app layout
st.title("Interactive Single Pendulum Simulation")

# Sidebar for user inputs
st.sidebar.header("Pendulum Parameters")
L = st.sidebar.slider("Pendulum Length (m)", 0.1, 2.0, 1.0, 0.1)
theta0 = st.sidebar.slider("Initial Angle (degrees)", 0, 90, 45, 5)
t_max = st.sidebar.slider("Simulation Time (s)", 5.0, 20.0, 10.0, 1.0)
dt = 0.01  # Fixed time step for simplicity

# Pendulum simulation function
def simulate_pendulum(L, theta0, t_max, dt):
    g = 9.81  # Gravitational acceleration (m/s^2)
    theta0_rad = np.radians(theta0)  # Convert initial angle to radians
    omega0 = 0.0  # Initial angular velocity (rad/s)
    
    # Time array
    t = np.arange(0, t_max, dt)
    n = len(t)
    
    # Arrays for angular displacement and velocity
    theta = np.zeros(n)
    omega = np.zeros(n)
    theta[0] = theta0_rad
    omega[0] = omega0
    
    # Euler method integration
    for i in range(n - 1):
        alpha = -(g / L) * np.sin(theta[i])  # Angular acceleration
        omega[i + 1] = omega[i] + alpha * dt
        theta[i + 1] = theta[i] + omega[i + 1] * dt
    
    # Convert to Cartesian coordinates
    x = L * np.sin(theta)
    y = -L * np.cos(theta)
    
    return t, np.degrees(theta), x, y

# Run simulation when button is clicked
if st.button("Run Simulation"):
    # Simulate pendulum
    t, theta_deg, x, y = simulate_pendulum(L, theta0, t_max, dt)
    
    # Plot 1: Angular displacement vs time
    fig1 = px.line(x=t, y=theta_deg, labels={"x": "Time (s)", "y": "Angular Displacement (degrees)"})
    fig1.update_layout(title="Pendulum Angle over Time", showlegend=False)
    fig1.update_traces(line_color="blue")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Plot 2: Pendulum trajectory
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Trajectory", line=dict(color="blue")))
    fig2.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="Pivot", marker=dict(size=10, color="black")))
    fig2.update_layout(
        title="Pendulum Trajectory",
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        xaxis_range=[-L - 0.1, L + 0.1],
        yaxis_range=[-L - 0.1, 0.1],
        showlegend=True,
        aspectratio=dict(x=1, y=1)
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Animation: Pendulum motion
    fig3 = go.Figure()
    # Initial frame
    fig3.add_trace(go.Scatter(
        x=[0, x[0]], y=[0, y[0]],
        mode="lines+markers",
        name="Pendulum",
        line=dict(color="blue", width=2),
        marker=dict(size=10)
    ))
    # Animation frames (decimated for performance)
    frames = [go.Frame(data=[go.Scatter(
        x=[0, x[k]], y=[0, y[k]],
        mode="lines+markers"
    )]) for k in range(0, len(t), 5)]
    fig3.frames = frames
    # Layout with animation controls
    fig3.update_layout(
        title="Pendulum Animation",
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        xaxis_range=[-L - 0.1, L + 0.1],
        yaxis_range=[-L - 0.1, 0.1],
        showlegend=False,
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}],
                 "label": "Play", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                 "label": "Pause", "method": "animate"}
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        aspectratio=dict(x=1, y=1)
    )
    st.plotly_chart(fig3, use_container_width=True)
