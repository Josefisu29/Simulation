import streamlit as st
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go

# Define the differential equations for the double pendulum
def double_pendulum_deriv(state, t, m1, m2, l1, l2, g):
    """
    Compute the derivatives of the double pendulum system.
    state: [theta1, omega1, theta2, omega2] (angles and angular velocities)
    t: time
    m1, m2: masses of the pendulums
    l1, l2: lengths of the pendulums
    g: gravitational acceleration
    """
    theta1, omega1, theta2, omega2 = state
    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)
    
    denominator = (2 * m1 + m2 - m2 * c**2)
    dtheta1 = omega1
    domega1 = (-g * (2 * m1 + m2) * np.sin(theta1) - m2 * g * np.sin(theta1 - 2 * theta2) - 
               2 * s * m2 * (omega2**2 * l2 + omega1**2 * l1 * c)) / (l1 * denominator)
    dtheta2 = omega2
    domega2 = (2 * s * (omega1**2 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(theta1) + 
                        omega2**2 * l2 * m2 * c)) / (l2 * denominator)
    
    return [dtheta1, domega1, dtheta2, domega2]

# Set up the Streamlit app
st.title("Interactive Double Pendulum Simulation")

# Sidebar for parameter inputs
st.sidebar.header("Parameters")
l1 = st.sidebar.slider("Length 1 (m)", 0.1, 2.0, 1.0, 0.1)
l2 = st.sidebar.slider("Length 2 (m)", 0.1, 2.0, 1.0, 0.1)
m1 = st.sidebar.slider("Mass 1 (kg)", 0.1, 2.0, 1.0, 0.1)
m2 = st.sidebar.slider("Mass 2 (kg)", 0.1, 2.0, 1.0, 0.1)
theta1_init = st.sidebar.slider("Initial Angle 1 (deg)", -90, 90, 45, 5)
theta2_init = st.sidebar.slider("Initial Angle 2 (deg)", -90, 90, 0, 5)
t_max = st.sidebar.slider("Simulation Time (s)", 5, 30, 10, 1)
dt = 0.01  # Time step for simulation

# Convert initial angles from degrees to radians
theta1_init_rad = np.radians(theta1_init)
theta2_init_rad = np.radians(theta2_init)

# Initial state: [theta1, omega1, theta2, omega2]
state0 = [theta1_init_rad, 0, theta2_init_rad, 0]

# Time array for the simulation
t = np.arange(0, t_max, dt)

# Button to trigger the simulation
if st.button("Run Simulation"):
    # Solve the differential equations
    solution = odeint(double_pendulum_deriv, state0, t, args=(m1, m2, l1, l2, 9.81))
    theta1, theta2 = solution[:, 0], solution[:, 2]
    
    # Compute Cartesian coordinates of the pendulum bobs
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)
    
    # Create the Plotly figure
    fig = go.Figure()
    
    # Initial frame (t=0)
    fig.add_trace(go.Scatter(
        x=[0, x1[0], x2[0]], 
        y=[0, y1[0], y2[0]], 
        mode='lines+markers', 
        name='Pendulum',
        line=dict(color='blue'),
        marker=dict(size=10)
    ))
    
    # Animation frames (decimated for performance)
    frames = [go.Frame(data=[go.Scatter(
        x=[0, x1[k], x2[k]], 
        y=[0, y1[k], y2[k]], 
        mode='lines+markers'
    )]) for k in range(0, len(t), 5)]
    fig.frames = frames
    
    # Update layout with animation controls
    fig.update_layout(
        xaxis_range=[-(l1 + l2) - 0.1, (l1 + l2) + 0.1],
        yaxis_range=[-(l1 + l2) - 0.1, 0.1],
        title="Double Pendulum Simulation",
        xaxis_title="X Position (m)",
        yaxis_title="Y Position (m)",
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}],
                 "label": "Play", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
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
        }]
    )
    
    # Display the animated plot
    st.plotly_chart(fig)
