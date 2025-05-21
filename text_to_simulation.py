import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import io
import json
import os
import requests
import streamlit.components.v1 as components # Added for Three.js

# Only import dotenv if not running on Streamlit Cloud
if not hasattr(st.secrets, "VERTEX_AI_API_KEY"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

# Get API key from st.secrets (Streamlit Cloud) or .env (local)
VERTEX_AI_API_KEY = st.secrets["VERTEX_AI_API_KEY"] if "VERTEX_AI_API_KEY" in st.secrets else os.getenv("VERTEX_AI_API_KEY")

# Ensure google.auth and related libraries are available for get_access_token_from_service_account
try:
    import google.auth
    import google.auth.transport.requests
    from google.oauth2 import service_account
except ImportError:
    st.error("google-auth and google-auth-oauthlib are required for Service Account authentication. Please install them.")
    # Optionally, provide installation instructions or halt execution
    # For example: st.stop()

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
        return None, None, None
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
    while current_time <= duration:
        time_data.append(current_time)
        position_data.append(pos)
        velocity_data.append(vel)
        pos = pos + vel * time_step + 0.5 * acc * (time_step ** 2)
        vel = vel + acc * time_step
        current_time += time_step
        if current_time > duration and (current_time - time_step) < duration:
            current_time = duration
    return time_data, position_data, velocity_data

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

# --- Function to create Three.js HTML ---
def create_threejs_visualization_html():
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Three.js Parameter Visualization</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <style>
            body {{ margin: 0; overflow: hidden; font-family: 'Inter', sans-serif; }}
            /* Adjusted to be relative to the container div, not 100vw/vh */
            #threejs_canvas_container {{ width: 100%; height: 100%; display: block; }} 
            #controls {{
                position: absolute;
                top: 10px; /* Adjusted for less padding if container is smaller */
                left: 10px;
                background-color: rgba(30, 41, 59, 0.9); /* bg-slate-800 with opacity */
                padding: 15px; /* Slightly reduced padding */
                border-radius: 12px;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                color: white;
                max-width: 280px; /* Adjusted max-width */
                max-height: calc(100% - 20px); /* Relative to parent height */
                overflow-y: auto;
                font-size: 0.8rem; /* Smaller base font for controls */
            }}
            #controls::-webkit-scrollbar {{
                width: 6px;
            }}
            #controls::-webkit-scrollbar-track {{
                background: #2d3748; /* bg-slate-700 */
                border-radius: 10px;
            }}
            #controls::-webkit-scrollbar-thumb {{
                background: #4a5568; /* bg-slate-600 */
                border-radius: 10px;
            }}
            #controls::-webkit-scrollbar-thumb:hover {{
                background: #718096; /* bg-slate-500 */
            }}
            .control-group {{ margin-bottom: 10px; }}
            .control-group label {{
                display: block;
                margin-bottom: 4px;
                font-weight: 500;
                color: #cbd5e1; /* text-slate-300 */
            }}
            .control-group input[type="range"], .control-group input[type="color"], .control-group select {{
                width: 100%;
                margin-top: 4px;
                border-radius: 6px;
                font-size: 0.8rem;
            }}
            .control-group input[type="range"] {{
                accent-color: #3b82f6; /* accent-blue-500 */
            }}
            .control-group input[type="color"] {{
                height: 30px;
                padding: 2px;
                border: 1px solid #4a5568; /* border-slate-600 */
                background-clip: content-box; 
            }}
             .control-group select {{
                background-color: #374151; /* bg-gray-700 for select */
                border: 1px solid #4a5568;
                padding: 5px;
            }}
            .control-group .value-display {{
                font-size: 0.75rem; /* Smaller value display */
                color: #94a3b8; /* text-slate-400 */
                margin-left: 5px;
            }}
            h2 {{
                font-size: 1.1rem; /* Smaller heading */
                font-weight: 600;
                margin-bottom: 0.8rem;
                border-bottom: 1px solid #4a5568; /* border-slate-600 */
                padding-bottom: 0.4rem;
            }}
        </style>
    </head>
    <body>
        <!-- Renamed outer container to avoid conflict if user has #container elsewhere -->
        <div id="threejs_visualization_app_container" style="width: 100%; height: 100%; position: relative;">
            <div id="threejs_canvas_container"></div> <!-- Canvas will be appended here -->
            <div id="controls"> <!-- Controls are absolutely positioned relative to this container -->
                <h2>Controls</h2>

                <div class="control-group">
                    <label for="shape">Shape:</label>
                    <select id="shape" class="w-full p-2 rounded-md bg-slate-700 border border-slate-600 focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                        <option value="cube">Cube</option>
                        <option value="sphere">Sphere</option>
                        <option value="torus">Torus</option>
                        <option value="cone">Cone</option>
                    </select>
                </div>

                <div class="control-group">
                    <label>Position:</label>
                    <div>X: <input type="range" id="posX" min="-5" max="5" value="0" step="0.1"><span class="value-display" id="posXVal">0</span></div>
                    <div>Y: <input type="range" id="posY" min="-5" max="5" value="0" step="0.1"><span class="value-display" id="posYVal">0</span></div>
                    <div>Z: <input type="range" id="posZ" min="-5" max="5" value="0" step="0.1"><span class="value-display" id="posZVal">0</span></div>
                </div>

                <div class="control-group">
                    <label>Rotation (degrees):</label>
                    <div>X: <input type="range" id="rotX" min="0" max="360" value="0" step="1"><span class="value-display" id="rotXVal">0</span></div>
                    <div>Y: <input type="range" id="rotY" min="0" max="360" value="0" step="1"><span class="value-display" id="rotYVal">0</span></div>
                    <div>Z: <input type="range" id="rotZ" min="0" max="360" value="0" step="1"><span class="value-display" id="rotZVal">0</span></div>
                </div>

                <div class="control-group">
                    <label for="scale">Scale:</label>
                    <input type="range" id="scale" min="0.1" max="3" value="1" step="0.05"><span class="value-display" id="scaleVal">1</span>
                </div>

                <div class="control-group">
                    <label for="color">Color:</label>
                    <input type="color" id="color" value="#3b82f6">
                </div>

                <div class="control-group">
                    <label for="lightIntensity">Light Intensity:</label>
                    <input type="range" id="lightIntensity" min="0" max="5" value="1.5" step="0.1"><span class="value-display" id="lightIntensityVal">1.5</span>
                </div>

                <div class="control-group">
                    <label for="animationSpeed">Animation Speed (Y-axis):</label>
                    <input type="range" id="animationSpeed" min="0" max="0.1" value="0.005" step="0.001"><span class="value-display" id="animationSpeedVal">0.005</span>
                </div>
            </div>
        </div>

        <script>
            let scene, camera, renderer, object, ambientLight, directionalLight;
            let currentShape = 'cube';

            // DOM Elements for controls
            const shapeSelector = document.getElementById('shape');
            const posXSlider = document.getElementById('posX');
            const posYSlider = document.getElementById('posY');
            const posZSlider = document.getElementById('posZ');
            const rotXSlider = document.getElementById('rotX');
            const rotYSlider = document.getElementById('rotY');
            const rotZSlider = document.getElementById('rotZ');
            const scaleSlider = document.getElementById('scale');
            const colorPicker = document.getElementById('color');
            const lightIntensitySlider = document.getElementById('lightIntensity');
            const animationSpeedSlider = document.getElementById('animationSpeed');

            const posXVal = document.getElementById('posXVal');
            const posYVal = document.getElementById('posYVal');
            const posZVal = document.getElementById('posZVal');
            const rotXVal = document.getElementById('rotXVal');
            const rotYVal = document.getElementById('rotYVal');
            const rotZVal = document.getElementById('rotZVal');
            const scaleVal = document.getElementById('scaleVal');
            const lightIntensityVal = document.getElementById('lightIntensityVal');
            const animationSpeedVal = document.getElementById('animationSpeedVal');
            
            const canvasContainer = document.getElementById('threejs_canvas_container'); // Target for renderer

            function init() {{
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x1e293b); 

                camera = new THREE.PerspectiveCamera(75, canvasContainer.clientWidth / canvasContainer.clientHeight, 0.1, 1000);
                camera.position.z = 5;

                renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
                canvasContainer.appendChild(renderer.domElement); // Append to specific div

                ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
                scene.add(ambientLight);

                directionalLight = new THREE.DirectionalLight(0xffffff, 1);
                directionalLight.position.set(5, 10, 7.5);
                scene.add(directionalLight);

                createObject(currentShape);

                shapeSelector.addEventListener('change', onShapeChange);
                posXSlider.addEventListener('input', updateObjectProperties);
                posYSlider.addEventListener('input', updateObjectProperties);
                posZSlider.addEventListener('input', updateObjectProperties);
                rotXSlider.addEventListener('input', updateObjectProperties);
                rotYSlider.addEventListener('input', updateObjectProperties);
                rotZSlider.addEventListener('input', updateObjectProperties);
                scaleSlider.addEventListener('input', updateObjectProperties);
                colorPicker.addEventListener('input', updateObjectProperties);
                lightIntensitySlider.addEventListener('input', updateLightProperties);
                animationSpeedSlider.addEventListener('input', updateAnimationProperties);

                updateValueDisplays();
                window.addEventListener('resize', onWindowResize, false);
                animate();
            }}

            function createObject(shapeType) {{
                if (object) {{
                    scene.remove(object);
                    object.geometry.dispose();
                    object.material.dispose();
                }}
                let geometry;
                switch (shapeType) {{
                    case 'sphere': geometry = new THREE.SphereGeometry(1, 32, 32); break;
                    case 'torus': geometry = new THREE.TorusGeometry(1, 0.4, 16, 100); break;
                    case 'cone': geometry = new THREE.ConeGeometry(1, 2, 32); break;
                    case 'cube': default: geometry = new THREE.BoxGeometry(1.5, 1.5, 1.5); break;
                }}
                const material = new THREE.MeshStandardMaterial({{
                    color: parseInt(colorPicker.value.replace('#', '0x')),
                    roughness: 0.5, metalness: 0.5
                }});
                object = new THREE.Mesh(geometry, material);
                scene.add(object);
                updateObjectProperties();
            }}

            function onShapeChange(event) {{
                currentShape = event.target.value;
                createObject(currentShape);
            }}

            function updateObjectProperties() {{
                if (!object) return;
                object.position.x = parseFloat(posXSlider.value);
                object.position.y = parseFloat(posYSlider.value);
                object.position.z = parseFloat(posZSlider.value);
                object.rotation.x = THREE.MathUtils.degToRad(parseFloat(rotXSlider.value));
                object.rotation.y = THREE.MathUtils.degToRad(parseFloat(rotYSlider.value));
                object.rotation.z = THREE.MathUtils.degToRad(parseFloat(rotZSlider.value));
                const scaleValue = parseFloat(scaleSlider.value);
                object.scale.set(scaleValue, scaleValue, scaleValue);
                object.material.color.set(colorPicker.value);
                updateValueDisplays();
            }}

            function updateLightProperties() {{
                const intensity = parseFloat(lightIntensitySlider.value);
                directionalLight.intensity = intensity;
                updateValueDisplays();
            }}

            function updateAnimationProperties() {{ updateValueDisplays(); }}

            function updateValueDisplays() {{
                posXVal.textContent = posXSlider.value;
                posYVal.textContent = posYSlider.value;
                posZVal.textContent = posZSlider.value;
                rotXVal.textContent = rotXSlider.value;
                rotYVal.textContent = rotYSlider.value;
                rotZVal.textContent = rotZSlider.value;
                scaleVal.textContent = scaleSlider.value;
                lightIntensityVal.textContent = lightIntensitySlider.value;
                animationSpeedVal.textContent = animationSpeedSlider.value;
            }}

            function onWindowResize() {{
                if (canvasContainer.clientWidth > 0 && canvasContainer.clientHeight > 0) {{
                    camera.aspect = canvasContainer.clientWidth / canvasContainer.clientHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
                }}
            }}

            function animate() {{
                requestAnimationFrame(animate);
                if (object) {{
                    const speed = parseFloat(animationSpeedSlider.value);
                    object.rotation.y += speed;
                }}
                renderer.render(scene, camera);
            }}
            
            // Ensure DOM is ready before init
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', init);
            }} else {{
                init(); // DOMContentLoaded has already fired
            }}
        </script>
    </body>
    </html>
    """
    return html_content

# --- COST CALCULATION CONSTANTS (EXAMPLE - REPLACE WITH ACTUAL CURRENT PRICING!) ---
MODEL_PRICING = {
    "gemini-1.0-pro-002": { # Updated model identifier
        "input_cost_per_1k_chars": 0.000125, # USD
        "output_cost_per_1k_chars": 0.000375, # USD
        "currency": "USD"
    },
    "gemini-1.5-flash-001": { # Ensuring this one is also specific
        "input_cost_per_1k_chars": 0.0000625, # USD
        "output_cost_per_1k_chars": 0.0001875, # USD
        "currency": "USD"
    }
}

USD_TO_INR_RATE = 83.0 # Placeholder exchange rate: 1 USD = 83 INR

# --- Cost Estimation Function ---
def estimate_vertex_ai_cost(model_identifier, input_chars, output_chars):
    pricing_info = MODEL_PRICING.get(model_identifier)
    if not pricing_info:
        return "Cost unknown (pricing info not found for this model)", 0.0, 0.0
    input_cost_usd = (input_chars / 1000) * pricing_info["input_cost_per_1k_chars"]
    output_cost_usd = (output_chars / 1000) * pricing_info["output_cost_per_1k_chars"]
    total_cost_usd = input_cost_usd + output_cost_usd
    currency = pricing_info["currency"]
    total_cost_inr = total_cost_usd * USD_TO_INR_RATE
    return f"Estimated Cost: {total_cost_usd:.6f} {currency} (~ {total_cost_inr:.2f} INR)", total_cost_usd, total_cost_inr

# --- Vertex AI REST Call Function (Service Account OAuth2) ---
def get_access_token_from_service_account(sa_json_path):
    """
    Returns an OAuth2 access token using a Google service account JSON key file.
    """
    import google.auth
    import google.auth.transport.requests
    from google.oauth2 import service_account
    SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
    credentials = service_account.Credentials.from_service_account_file(sa_json_path, scopes=SCOPES)
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    return credentials.token

# --- Vertex AI REST Call Function ---
def get_simulation_parameters_from_vertex_ai(text_prompt, model_identifier, project, location, sa_json_path=None):
    """
    Calls Vertex AI Gemini model using REST API and OAuth2 access token from service account JSON.
    """
    input_character_count = len(text_prompt)
    output_character_count = 0
    if not sa_json_path or not os.path.exists(sa_json_path):
        return None, input_character_count, 0, "Service account JSON key file not found. Please upload it and provide the path."
    try:
        access_token = get_access_token_from_service_account(sa_json_path)
    except Exception as e:
        return None, input_character_count, 0, f"Failed to get access token: {e}"
    endpoint = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model_identifier}:predict"
    headers = {
        "Authorization": f"Bearer {access_token}",
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
input_cost_usd_sidebar = MODEL_PRICING[selected_model_id]['input_cost_per_1k_chars']
output_cost_usd_sidebar = MODEL_PRICING[selected_model_id]['output_cost_per_1k_chars']
input_cost_inr_sidebar = input_cost_usd_sidebar * USD_TO_INR_RATE
output_cost_inr_sidebar = output_cost_usd_sidebar * USD_TO_INR_RATE

st.sidebar.markdown(f"""
**Selected Model Pricing ({MODEL_PRICING[selected_model_id]['currency']}/1k chars):**
- Input: {input_cost_usd_sidebar:.6f} USD (~ {input_cost_inr_sidebar:.4f} INR)
- Output: {output_cost_usd_sidebar:.6f} USD (~ {output_cost_inr_sidebar:.4f} INR)
""")

# --- Cost Calculator in Main UI ---
st.markdown("---")
st.markdown("### 💲 Vertex AI Cost Calculator (Estimate)")
user_prompt_for_cost = st.text_area("Enter your prompt to estimate cost:", "a ball moving at 10 m/s for 5 seconds", key="cost_prompt")
simulated_output = '{"object_name": "ball", "initial_position": 0, "velocity": 10, "acceleration": 0, "time_duration": 5, "time_step": 0.1}'
input_chars = len(user_prompt_for_cost)
output_chars = len(simulated_output)
if st.button("Estimate Vertex AI API Cost"):
    cost_message, total_cost_usd, total_cost_inr = estimate_vertex_ai_cost(selected_model_id, input_chars, output_chars)
    st.info(f"{cost_message}\n(Input: {input_chars} chars, Output: {output_chars} chars)")

# --- Live Vertex AI Call UI ---
st.markdown("---")
st.markdown("### 🤖 Run Live Vertex AI Simulation Parameter Extraction")
live_prompt = st.text_area("Enter your simulation description for live Vertex AI call:", "a ball moving at 10 m/s for 5 seconds", key="live_vertex_prompt")
# Add file uploader for service account JSON key
sa_json_file = st.file_uploader("Upload Service Account JSON Key File", type=["json"], key="sa_json_file")
sa_json_path = None
if sa_json_file is not None:
    # Save uploaded file to a temp location
    sa_json_path = f"temp_sa_{int(st.session_state.get('sa_file_counter', 0))}.json"
    with open(sa_json_path, "wb") as f:
        f.write(sa_json_file.read())
    st.session_state['sa_file_counter'] = st.session_state.get('sa_file_counter', 0) + 1

if st.button("Call Vertex AI (Gemini) Live"):
    if not sa_json_path:
        st.error("Please upload a service account JSON key file.")
    else:
        st.write(f"Attempting to call API with model ID: '{selected_model_id}'") # DEBUG LINE
        with st.spinner(f"Calling Vertex AI model {selected_model_id}..."):
            response_text, input_chars, output_chars, error = get_simulation_parameters_from_vertex_ai(
                live_prompt, selected_model_id, PROJECT_ID, LOCATION, sa_json_path
            )
        if error:
            st.error(f"Vertex AI Error: {error}")
        elif response_text:
            st.success("Vertex AI response received!")
            st.markdown("**Raw Model Output:**")
            st.code(response_text, language="json")
            cost_message, total_cost_usd, total_cost_inr = estimate_vertex_ai_cost(selected_model_id, input_chars, output_chars)
            st.info(f"{cost_message}\n(Input: {input_chars} chars, Output: {output_chars} chars)")
            # Try to parse JSON for downstream simulation
            try:
                sim_params = json.loads(response_text)
                time_data, position_data, velocity_data = run_simulation_and_generate_data(sim_params)
                time_data = time_data if time_data is not None else []
                position_data = position_data if position_data is not None else []
                velocity_data = velocity_data if velocity_data is not None else []
                obj = sim_params.get('object_name', 'object')
                acc = sim_params.get('acceleration', 0.0)
                velocity = sim_params.get('velocity', 0.0)
                
                st.success(f"Simulation for **{obj.capitalize()}** completed based on extracted parameters.")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 2D Plot Visualization")
                    if time_data and position_data and velocity_data:
                        fig = plot_simulation(time_data, position_data, velocity_data, obj, acc, velocity)
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.info("Not enough data for 2D plot.")

                with col2:
                    st.markdown("### 3D Interactive Shape Visualization")
                    threejs_html = create_threejs_visualization_html()
                    components.html(threejs_html, height=600, scrolling=False)
                
                st.markdown("---")
                st.markdown("#### Download Data")
                if time_data and position_data and velocity_data:
                    csv = "Time,Position,Velocity\n" + "\n".join([f"{t},{p},{v}" for t,p,v in zip(list(time_data), list(position_data), list(velocity_data))])
                    st.download_button("Download CSV", csv, file_name=f"{obj}_simulation.csv", mime="text/csv")
                else:
                    st.info("No data to download.")

            except Exception as e:
                st.warning(f"Could not process or visualize simulation results: {e}")
        else:
            st.warning("No output received from Vertex AI.")

with st.form("sim_form", clear_on_submit=False):
    user_prompt = st.text_input("Describe the simulation:", "a ball moving at 10 m/s for 5 seconds")
    submitted = st.form_submit_button("Simulate & Visualize")

if submitted:
    simulation_params = call_gemini_api(user_prompt)
    if simulation_params:
        time_data, position_data, velocity_data = run_simulation_and_generate_data(simulation_params)
        obj = simulation_params.get('object_name', 'object')
        acc = simulation_params.get('acceleration', 0.0)
        velocity = simulation_params.get('velocity', 0.0)

        time_data = time_data if time_data is not None else []
        position_data = position_data if position_data is not None else []
        velocity_data = velocity_data if velocity_data is not None else []

        st.success(f"Simulation for **{obj.capitalize()}** completed using placeholder data.")
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 2D Plot Visualization")
            if time_data and position_data and velocity_data:
                fig = plot_simulation(time_data, position_data, velocity_data, obj, acc, velocity)
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Not enough data for 2D plot.")

        with col2:
            st.markdown("### 3D Interactive Shape Visualization")
            threejs_html = create_threejs_visualization_html()
            components.html(threejs_html, height=600, scrolling=False)
        
        st.markdown("---")
        st.markdown("#### Download Data")
        if time_data and position_data and velocity_data:
            csv = "Time,Position,Velocity\n" + "\n".join([f"{t},{p},{v}" for t,p,v in zip(list(time_data), list(position_data), list(velocity_data))])
            st.download_button("Download CSV", csv, file_name=f"{obj}_simulation.csv", mime="text/csv")
        else:
            st.info("No data to download.")
    else:
        st.error("Could not understand the prompt. Please try a different description.")
