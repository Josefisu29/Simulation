import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Streamlit app layout
st.title("Interactive Pendulum Simulation")

# Sidebar for user inputs
st.sidebar.header("Pendulum Parameters")
length_type = st.sidebar.radio("Length Type", ["Fixed", "Dynamic"])

if length_type == "Fixed":
    length_cm = st.sidebar.selectbox("Select pendulum length (cm)", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    L = length_cm / 100.0  # Convert cm to meters
    st.sidebar.write(f"Selected length: {length_cm} cm ({L:.2f} m)")
else:
    L_start = st.sidebar.slider("Start Length (m)", 0.1, 2.0, 1.0, 0.1)
    L_end = st.sidebar.slider("End Length (m)", 0.1, 2.0, 1.5, 0.1)

theta0 = st.sidebar.slider("Initial Angle (degrees)", 0, 90, 45, 5)
t_max = st.sidebar.slider("Total Simulation Time (s)", 5.0, 20.0, 10.0, 1.0)
dt = 0.01  # Fixed time step

# Pendulum simulation function
def simulate_pendulum(L_array, theta0, t):
    g = 9.81  # Gravitational acceleration (m/s^2)
    theta0_rad = np.radians(theta0)
    n = len(t)
    dt = t[1] - t[0]
    theta = np.zeros(n)
    omega = np.zeros(n)
    theta[0] = theta0_rad
    omega[0] = 0.0
    
    # Euler method integration
    for i in range(n - 1):
        alpha = -(g / L_array[i]) * np.sin(theta[i])
        omega[i + 1] = omega[i] + alpha * dt
        theta[i + 1] = theta[i] + omega[i + 1] * dt
    
    theta_deg = np.degrees(theta)
    x = L_array * np.sin(theta)
    y = -L_array * np.cos(theta)
    
    # Oscillation analysis
    crossings = np.where(np.diff(np.sign(theta)))[0]
    num_oscillations = len(crossings) // 2
    if num_oscillations > 0:
        periods = np.diff(t[crossings])
        avg_period = np.mean(periods)
    else:
        avg_period = np.nan
    
    return t, theta_deg, x, y, num_oscillations, avg_period, L_array

# Run simulation when button is clicked
if st.button("Run Simulation"):
    t = np.arange(0, t_max, dt)
    if length_type == "Fixed":
        L_array = np.full(len(t), L)
    else:
        L_array = L_start + (L_end - L_start) * t / t_max
    
    t, theta_deg, x, y, num_oscillations, avg_period, L_array = simulate_pendulum(L_array, theta0, t)
    
    # Analysis text
    if length_type == "Fixed":
        T_theory = 2 * np.pi * np.sqrt(L / 9.81)
        analysis_text = f"Oscillations: {num_oscillations}<br>Avg Period: {avg_period:.2f} s<br>Theoretical Period: {T_theory:.2f} s"
        title_suffix = f"Fixed Length: {L:.2f} m"
    else:
        analysis_text = f"Oscillations: {num_oscillations}<br>Avg Period: {avg_period:.2f} s"
        title_suffix = f"Length varies from {L_start:.2f} to {L_end:.2f} m"
    
    # Plot 1: Angular displacement vs time
    fig1 = px.line(x=t, y=theta_deg, labels={"x": "Time (s)", "y": "Angular Displacement (degrees)"})
    fig1.update_layout(
        title=f"Pendulum Angle over Time ({title_suffix})",
        showlegend=False,
        annotations=[dict(x=0.05, y=0.95, xref="paper", yref="paper", text=analysis_text, showarrow=False, bgcolor="white", bordercolor="gray", borderwidth=1, align="left")]
    )
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
        xaxis=dict(range=[-np.max(L_array) - 0.1, np.max(L_array) + 0.1], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-np.max(L_array) - 0.1, 0.1]),
        showlegend=True
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Animation: Pendulum motion
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=[0, x[0]], y=[0, y[0]], mode="lines+markers", name="Pendulum", line=dict(color="blue", width=2), marker=dict(size=10)))
    frames = [go.Frame(data=[go.Scatter(x=[0, x[k]], y=[0, y[k]], mode="lines+markers")]) for k in range(0, len(t), 5)]
    fig3.frames = frames
    fig3.update_layout(
        title="Pendulum Animation",
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        xaxis=dict(range=[-np.max(L_array) - 0.1, np.max(L_array) + 0.1], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-np.max(L_array) - 0.1, 0.1]),
        showlegend=False,
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}], "label": "Play", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}], "label": "Pause", "method": "animate"}
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )
    st.plotly_chart(fig3, use_container_width=True)
