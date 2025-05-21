import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import io

# Set Streamlit page config for better mobile/desktop experience
st.set_page_config(page_title="Text-to-Simulation", layout="centered", initial_sidebar_state="auto")

# Custom CSS for border and responsive design
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .stTextInput>div>div>input {
        border: 2px solid #4F8BF9;
        border-radius: 8px;
        padding: 8px;
    }
    .stButton>button {
        border-radius: 8px;
        border: 2px solid #4F8BF9;
        background: #4F8BF9;
        color: white;
        font-weight: bold;
        padding: 8px 24px;
    }
    .stMarkdown, .stCodeBlock, .stDataFrame, .stTable {
        border-radius: 8px;
        border: 1.5px solid #e0e0e0;
        background: #fff;
        padding: 12px;
        margin-bottom: 16px;
    }
    @media (max-width: 600px) {
        .stApp {
            padding: 0 2vw;
        }
        .stTextInput>div>div>input, .stButton>button {
            font-size: 1.1em;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- BEGIN: User Guide and About Section ---
with st.expander("ℹ️ How to Use This App", expanded=True):
    st.markdown("""
    ### Welcome to **Text-to-Simulation**!
    This app lets you describe a simple physics scenario in plain English and instantly see a simulation and visualization.

    **How to Use:**
    1. **Describe your simulation** in the input box. Example prompts:
        - `a ball moving at 10 m/s for 5 seconds`
        - `rock falling for 3 seconds from 100m`
        - `car accelerates at 2 m/s^2 for 10s`
    2. **Click 'Simulate & Visualize'** to run the simulation.
    3. **View the simulation log** to see how the object's position and velocity change over time.
    4. **See the visualization**: The app will plot position and velocity vs. time.
    5. **Download the data** as a CSV for your own analysis.

    **Tips:**
    - Use clear, simple English for best results.
    - The app currently supports basic 1D motion and simple acceleration scenarios.
    - If your prompt isn't understood, try rephrasing it.
    """)

with st.expander("🛠️ About the Tools & Technologies", expanded=False):
    st.markdown("""
    **Technologies Used:**
    - [Streamlit](https://streamlit.io): For building the interactive web interface. Streamlit makes it easy to create data apps with Python.
    - [Matplotlib](https://matplotlib.org): For plotting the simulation results (position and velocity over time).
    - **Python**: The core simulation logic is written in Python, using basic kinematic equations.
    - **Gemini API (Placeholder)**: In a real deployment, a language model API (like Gemini or similar) would extract simulation parameters from your text. Here, we use a simple placeholder for demonstration.

    **How it Works:**
    1. You enter a natural language description.
    2. The app (eventually via Gemini API) extracts the physical parameters (object, velocity, acceleration, etc.).
    3. The simulation engine computes the object's motion step by step.
    4. The results are displayed as a log and a plot.
    5. You can download the data for further use.

    **Responsive Design:**
    - The app uses custom CSS to ensure borders, padding, and layout look good on both desktop and mobile devices.
    """)
# --- END: User Guide and About Section ---

# Placeholder for Gemini API Integration
def call_gemini_api(text_prompt):
    if "ball moving at 10 m/s for 5 seconds" in text_prompt.lower():
        return {'object_name': 'ball', 'initial_position': 0, 'velocity': 10, 'acceleration': 0, 'time_duration': 5, 'time_step': 0.1}
    elif "rock falling for 3 seconds from 100m" in text_prompt.lower():
        return {'object_name': 'rock', 'initial_position': 100, 'initial_velocity': 0, 'acceleration': -9.81, 'time_duration': 3, 'time_step': 0.1}
    elif "car accelerates at 2 m/s^2 for 10s" in text_prompt.lower():
        return {'object_name': 'car', 'initial_position': 0, 'initial_velocity': 0, 'acceleration': 2, 'time_duration': 10, 'time_step': 0.1}
    else:
        return None

def run_simulation_and_generate_data(parameters):
    if not parameters:
        return None, None, None, None
    obj = parameters.get('object_name', 'object')
    pos = parameters.get('initial_position', 0.0)
    vel = parameters.get('initial_velocity', parameters.get('velocity', 0.0))
    acc = parameters.get('acceleration', 0.0)
    duration = parameters.get('time_duration', 1.0)
    time_step = parameters.get('time_step', 0.1)
    current_time = 0.0
    time_data = []
    position_data = []
    velocity_data = []
    log_lines = []
    while current_time <= duration:
        log_lines.append(f"{current_time:>8.2f} | {pos:>12.2f} | {vel:>14.2f}")
        time_data.append(current_time)
        position_data.append(pos)
        velocity_data.append(vel)
        pos = pos + vel * time_step + 0.5 * acc * (time_step ** 2)
        vel = vel + acc * time_step
        current_time += time_step
        if current_time > duration and (current_time - time_step) < duration:
            current_time = duration
    return time_data, position_data, velocity_data, log_lines

def plot_simulation(time_data, position_data, velocity_data, obj, acc, velocity):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)', color=color)
    ax1.plot(time_data, position_data, color=color, linestyle='-', marker='o', label=f'{obj} Position')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Velocity (m/s)', color=color)
    ax2.plot(time_data, velocity_data, color=color, linestyle='--', marker='x', label=f'{obj} Velocity')
    ax2.tick_params(axis='y', labelcolor=color)
    plot_title = f"Simulation of {obj.capitalize()}"
    if acc != 0:
        plot_title += f" (Acceleration: {acc} m/s^2)"
    else:
        plot_title += f" (Constant Velocity: {velocity} m/s)"
    fig.suptitle(plot_title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig

# Streamlit UI
st.title("🚀 Text-to-Simulation with Visualization")
st.markdown("""
Enter a natural language description of a simple physics simulation (e.g.,
- `a ball moving at 10 m/s for 5 seconds`
- `rock falling for 3 seconds from 100m`
- `car accelerates at 2 m/s^2 for 10s`
""")

with st.form("sim_form", clear_on_submit=False):
    user_prompt = st.text_input("Describe the simulation:", "a ball moving at 10 m/s for 5 seconds")
    submitted = st.form_submit_button("Simulate & Visualize")

if submitted:
    simulation_params = call_gemini_api(user_prompt)
    if simulation_params:
        time_data, position_data, velocity_data, log_lines = run_simulation_and_generate_data(simulation_params)
        obj = simulation_params.get('object_name', 'object')
        acc = simulation_params.get('acceleration', 0.0)
        velocity = simulation_params.get('velocity', 0.0)

        # Defensive: If any of the lists are None, replace with empty list
        time_data = time_data if time_data is not None else []
        position_data = position_data if position_data is not None else []
        velocity_data = velocity_data if velocity_data is not None else []
        log_lines = log_lines if log_lines is not None else []

        st.success(f"Simulation for **{obj.capitalize()}** completed.")
        st.markdown("### Simulation Log")
        st.code("Time (s) | Position (m) | Velocity (m/s)\n" + "-"*40 + "\n" + "\n".join([str(line) for line in log_lines]), language="text")
        st.markdown("### Visualization")
        fig = plot_simulation(time_data, position_data, velocity_data, obj, acc, velocity)
        st.pyplot(fig, use_container_width=True)
        st.markdown("---")
        st.markdown("#### Download Data")
        csv = "Time,Position,Velocity\n" + "\n".join([f"{t},{p},{v}" for t,p,v in zip(list(time_data), list(position_data), list(velocity_data))])
        st.download_button("Download CSV", csv, file_name=f"{obj}_simulation.csv", mime="text/csv")
    else:
        st.error("Could not understand the prompt. Please try a different description.")
