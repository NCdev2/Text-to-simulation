import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import io
import json
import os
import requests

# Only import dotenv if not running on Streamlit Cloud
if not hasattr(st.secrets, "VERTEX_AI_API_KEY"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

# Get API key from st.secrets (Streamlit Cloud) or .env (local)
VERTEX_AI_API_KEY = st.secrets["VERTEX_AI_API_KEY"] if "VERTEX_AI_API_KEY" in st.secrets else os.getenv("VERTEX_AI_API_KEY")

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
    - **Vertex AI & Machine Learning**: In a production system, Google Vertex AI can be used to host and serve advanced machine learning models (like Gemini or other LLMs). These models are trained to understand natural language and extract structured simulation parameters from your text prompt.

    **How Machine Learning & Vertex AI Enable Text-to-Simulation:**
    1. **Text Understanding**: When you enter a description, a large language model (LLM) hosted on Vertex AI analyzes your text and extracts key simulation parameters (object type, initial position, velocity, acceleration, duration, etc.).
    2. **Code Generation**: The extracted parameters are used to generate Python code that simulates the described scenario. This code can include both the simulation logic and the visualization (e.g., Matplotlib plotting code).
    3. **Preview & Visualisation**: The app can display the generated code and, if you choose, execute it to show a live plot of the simulation. This lets you see both the code and the resulting visualization, making the process transparent and interactive.

    **How it Works (End-to-End):**
    1. You enter a natural language description.
    2. The app (eventually via Gemini API or Vertex AI) extracts the physical parameters (object, velocity, acceleration, etc.) using machine learning.
    3. The simulation engine computes the object's motion step by step.
    4. The results are displayed as a log and a plot.
    5. You can download the data for further use.

    **Responsive Design:**
    - The app uses custom CSS to ensure borders, padding, and layout look good on both desktop and mobile devices.
    """)

with st.expander("💲 Vertex AI API Cost Estimation", expanded=False):
    st.markdown("""
    **Vertex AI API calls (for Gemini and similar models) are billed based on the amount of text you send (input) and receive (output), and the model you use.**

    - **Input cost:** Based on the number of characters in your prompt.
    - **Output cost:** Based on the number of characters in the model's response.
    - **Model:** Different Gemini models (e.g., Pro, Flash) have different prices.

    **How this app estimates cost:**
    1. You select a model and enter your prompt.
    2. The app counts the characters in your input and the (simulated) output.
    3. It multiplies these counts by the current per-1,000-character price for the selected model.
    4. The estimated cost is shown before you run the simulation.

    > **Note:** This is an estimate. Actual costs may vary. Always check the [official Vertex AI pricing page](https://cloud.google.com/vertex-ai/pricing) for the latest rates.
    """)

with st.expander("💡 Profit & Model Improvement: How Your Simulations Add Value", expanded=False):
    st.markdown("""
    ### How Profit is Generated
    - **Pay-per-use Model:** Each time you run a simulation using the Vertex AI API, a small fee is charged (see the cost calculator above). By offering this as a service, the platform can charge users per simulation, bundle credits, or offer premium features for a fee.
    - **Scalability:** As more users run simulations, the platform's revenue grows. Bulk or enterprise users can be offered volume discounts, while individual users pay per use or via subscription.
    - **Value-added Services:** Advanced analytics, downloadable reports, or integration with other tools can be offered as premium features, increasing profit potential.
    - **Data Insights:** Aggregated, anonymized usage data can inform new features, targeted offerings, or even be licensed (with user consent) for research or industry insights.

    ### How Your Usage Improves the AI
    - **User Feedback:** When you run a simulation, your feedback (e.g., Was the result accurate? Did the model understand your prompt?) can be collected (with your consent) to improve the underlying machine learning models.
    - **Prompt Diversity:** Every unique prompt helps the AI learn new ways people describe simulations, making the model more robust and accurate for everyone.
    - **Active Learning:** Difficult or ambiguous cases can be flagged for review, helping data scientists retrain and fine-tune the model for better future performance.
    - **Community-driven Improvement:** As more users interact, the system can identify gaps in understanding and expand its capabilities, benefiting all users.

    ### Why This Software is Best for Profit
    - **Low Overhead, High Scalability:** The cloud-based, API-driven approach means minimal infrastructure costs and easy scaling as user demand grows.
    - **Automated, Self-service:** Users can run simulations anytime, anywhere, without manual intervention, maximizing revenue potential.
    - **Continuous Model Improvement:** The more the platform is used, the smarter and more valuable it becomes, creating a virtuous cycle of improvement and profit.
    - **Transparent Costing:** Built-in cost calculators and clear pricing build user trust and encourage more usage.
    - **Unique Value Proposition:** By combining natural language, simulation, visualization, and cost transparency, this platform stands out from traditional simulation tools.
    """)
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

# --- COST CALCULATION CONSTANTS (EXAMPLE - REPLACE WITH ACTUAL CURRENT PRICING!) ---
MODEL_PRICING = {
    "gemini-1.0-pro-001": {
        "input_cost_per_1k_chars": 0.000125, # USD
        "output_cost_per_1k_chars": 0.000375, # USD
        "currency": "USD"
    },
    "gemini-1.5-flash-001": {
        "input_cost_per_1k_chars": 0.0000625, # USD
        "output_cost_per_1k_chars": 0.0001875, # USD
        "currency": "USD"
    }
}

# --- Cost Estimation Function ---
def estimate_vertex_ai_cost(model_identifier, input_chars, output_chars):
    pricing_info = MODEL_PRICING.get(model_identifier)
    if not pricing_info:
        return "Cost unknown (pricing info not found for this model)", 0.0
    input_cost = (input_chars / 1000) * pricing_info["input_cost_per_1k_chars"]
    output_cost = (output_chars / 1000) * pricing_info["output_cost_per_1k_chars"]
    total_cost = input_cost + output_cost
    currency = pricing_info["currency"]
    return f"Estimated Cost: {total_cost:.6f} {currency}", total_cost

# --- Vertex AI REST Call Function ---
def get_simulation_parameters_from_vertex_ai(text_prompt, model_identifier, project, location, credentials=None):
    """
    Calls Vertex AI Gemini model using REST API and API key from .env.
    """
    input_character_count = len(text_prompt)
    output_character_count = 0
    if not VERTEX_AI_API_KEY:
        return None, input_character_count, 0, "Vertex AI API key not found in .env file."
    endpoint = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model_identifier}:predict"
    headers = {
        "Authorization": f"Bearer {VERTEX_AI_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"""
    Extract the simulation parameters from the following text and return as a JSON object with keys: object_name, initial_position, initial_velocity, velocity, acceleration, time_duration, time_step.\nText: {text_prompt}
    """
    payload = {
        "instances": [{"content": prompt}],
        "parameters": {"temperature": 0.2, "maxOutputTokens": 256}
    }
    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            if result.get("predictions"):
                response_text = result["predictions"][0].get("content", "")
                output_character_count = len(response_text)
                return response_text, input_character_count, output_character_count, None
            else:
                return None, input_character_count, 0, "No content in response"
        else:
            return None, input_character_count, 0, f"Vertex AI REST error: {response.status_code} {response.text}"
    except Exception as e:
        return None, input_character_count, 0, str(e)

# Set your project and location for Vertex AI REST API
PROJECT_ID = "text-to-simulation"  # Set your GCP project ID here
LOCATION = "us-central1"            # Or your preferred region

# Streamlit UI
st.title("🚀 Text-to-Simulation with Visualization")
st.markdown("""
Enter a natural language description of a simple physics simulation (e.g.,
- `a ball moving at 10 m/s for 5 seconds`
- `rock falling for 3 seconds from 100m`
- `car accelerates at 2 m/s^2 for 10s`
""")

# --- Model Selection UI ---
available_models = list(MODEL_PRICING.keys())
selected_model_id = st.sidebar.selectbox("Select Vertex AI Model for Cost Estimate:", available_models)
st.sidebar.markdown(f"""
**Selected Model Pricing ({MODEL_PRICING[selected_model_id]['currency']}/1k chars):**
- Input: {MODEL_PRICING[selected_model_id]['input_cost_per_1k_chars']:.6f}
- Output: {MODEL_PRICING[selected_model_id]['output_cost_per_1k_chars']:.6f}
""")

# --- Cost Calculator in Main UI ---
st.markdown("---")
st.markdown("### 💲 Vertex AI Cost Calculator (Estimate)")
user_prompt_for_cost = st.text_area("Enter your prompt to estimate cost:", "a ball moving at 10 m/s for 5 seconds", key="cost_prompt")
simulated_output = '{"object_name": "ball", "initial_position": 0, "velocity": 10, "acceleration": 0, "time_duration": 5, "time_step": 0.1}'
input_chars = len(user_prompt_for_cost)
output_chars = len(simulated_output)
if st.button("Estimate Vertex AI API Cost"):
    cost_message, total_cost = estimate_vertex_ai_cost(selected_model_id, input_chars, output_chars)
    st.info(f"{cost_message}\n(Input: {input_chars} chars, Output: {output_chars} chars)")

# --- Live Vertex AI Call UI ---
st.markdown("---")
st.markdown("### 🤖 Run Live Vertex AI Simulation Parameter Extraction")
live_prompt = st.text_area("Enter your simulation description for live Vertex AI call:", "a ball moving at 10 m/s for 5 seconds", key="live_vertex_prompt")
if st.button("Call Vertex AI (Gemini) Live"):
    with st.spinner(f"Calling Vertex AI model {selected_model_id}..."):
        response_text, input_chars, output_chars, error = get_simulation_parameters_from_vertex_ai(
            live_prompt, selected_model_id, PROJECT_ID, LOCATION
        )
    if error:
        st.error(f"Vertex AI Error: {error}")
    elif response_text:
        st.success("Vertex AI response received!")
        st.markdown("**Raw Model Output:**")
        st.code(response_text, language="json")
        cost_message, total_cost = estimate_vertex_ai_cost(selected_model_id, input_chars, output_chars)
        st.info(f"{cost_message}\n(Input: {input_chars} chars, Output: {output_chars} chars)")
        # Try to parse JSON for downstream simulation
        try:
            # Defensive: If any of the lists are None, replace with empty list
            sim_params = json.loads(response_text)
            time_data, position_data, velocity_data, log_lines = run_simulation_and_generate_data(sim_params)
            time_data = time_data if time_data is not None else []
            position_data = position_data if position_data is not None else []
            velocity_data = velocity_data if velocity_data is not None else []
            log_lines = log_lines if log_lines is not None else []
            obj = sim_params.get('object_name', 'object')
            acc = sim_params.get('acceleration', 0.0)
            velocity = sim_params.get('velocity', 0.0)
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
        except Exception as e:
            st.warning(f"Could not parse model output as JSON: {e}")
    else:
        st.warning("No output received from Vertex AI.")

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
