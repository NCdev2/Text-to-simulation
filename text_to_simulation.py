import matplotlib
# Try to set a non-interactive backend if no display is available,
# or an interactive one if it is. This is a bit of a heuristic.
try:
    matplotlib.use('Agg') # For saving to file, non-interactive
    import matplotlib.pyplot as plt
except ImportError:
    plt = None # Matplotlib not installed
except Exception:
    plt = None

# Placeholder for Gemini API Integration
def call_gemini_api(text_prompt):
    """
    This function will interact with the Gemini API.
    It takes a natural language text prompt describing the simulation
    and is expected to return a structured representation of
    simulation parameters.

    Args:
        text_prompt (str): The user's description of the simulation.

    Returns:
        dict: A dictionary containing simulation parameters.
              e.g., {'object_name': 'ball', 'initial_position': 0, 'velocity': 10, 'acceleration': 0, 'time_duration': 5, 'time_step': 0.1}
              Returns None if the API call fails or parsing is unsuccessful.
    """
    print(f"\n[GEMINI API CALL (Placeholder)] Processing prompt: '{text_prompt}'")
    if "ball moving at 10 m/s for 5 seconds" in text_prompt.lower():
        print("[GEMINI API (Placeholder)] Understood: Ball, velocity 10 m/s, duration 5s.")
        return {'object_name': 'ball', 'initial_position': 0, 'velocity': 10, 'acceleration': 0, 'time_duration': 5, 'time_step': 0.1}
    elif "rock falling for 3 seconds from 100m" in text_prompt.lower():
        print("[GEMINI API (Placeholder)] Understood: Rock, gravity, duration 3s, initial height 100m.")
        return {'object_name': 'rock', 'initial_position': 100, 'initial_velocity': 0, 'acceleration': -9.81, 'time_duration': 3, 'time_step': 0.1}
    elif "car accelerates at 2 m/s^2 for 10s" in text_prompt.lower():
        print("[GEMINI API (Placeholder)] Understood: Car, acceleration 2 m/s^2, duration 10s.")
        return {'object_name': 'car', 'initial_position': 0, 'initial_velocity': 0, 'acceleration': 2, 'time_duration': 10, 'time_step': 0.1}
    else:
        print("[GEMINI API (Placeholder)] Could not understand the prompt fully. Using default parameters.")
        return None

# Simple Simulation Engine & Visualization Code Generator
def run_simulation_and_generate_code(parameters):
    """
    Runs a very basic physics simulation and generates Python code for visualization.
    Args:
        parameters (dict): A dictionary of simulation parameters.
    Returns:
        tuple: (simulation_log_string, visualization_code_string, can_preview)
               simulation_log_string: Text log of the simulation.
               visualization_code_string: Python code (string) for Matplotlib visualization.
               can_preview (bool): Whether a preview is possible (Matplotlib is available).
    """
    if not parameters:
        log = "\n[SIMULATION ENGINE] Error: No parameters provided for simulation."
        return log, None, False

    log_lines = ["\n[SIMULATION ENGINE] Starting simulation..."]
    log_lines.append(f"Parameters: {parameters}")

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

    header = f"\n--- {obj.capitalize()} Simulation Log ---"
    log_lines.append(header)
    log_lines.append(f"Time (s) | Position (m) | Velocity (m/s)")
    log_lines.append(f"---------|--------------|----------------")

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

    log_lines.append(f"---------|--------------|----------------")
    log_lines.append(f"[SIMULATION ENGINE] Simulation for {obj} finished after {duration:.2f} seconds.")
    simulation_log_string = "\n".join(log_lines)

    visualization_code = None
    can_preview_flag = bool(plt)

    if can_preview_flag:
        plot_title = f"Simulation of {obj.capitalize()}"
        if acc != 0:
            plot_title += f" (Acceleration: {acc} m/s^2)"
        else:
            plot_title += f" (Constant Velocity: {parameters.get('velocity', 0.0)} m/s)"

        visualization_code = f"""
# Visualization Code (Requires Matplotlib: pip install matplotlib)
import matplotlib.pyplot as plt

time = {time_data}
position = {position_data}
velocity = {velocity_data}
object_name = '{obj}'
plot_title = "{plot_title}"

fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:red'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Position (m)', color=color)
ax1.plot(time, position, color=color, linestyle='-', marker='o', label=f'{obj} Position')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Velocity (m/s)', color=color)
ax2.plot(time, velocity, color=color, linestyle='--', marker='x', label=f'{obj} Velocity')
ax2.tick_params(axis='y', labelcolor=color)

fig.suptitle(plot_title, fontsize=16)
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
plt.subplots_adjust(top=0.85)
plt.show()
# To save the figure:
# plt.savefig(f'{obj}_simulation_plot.png')
# print(f"Plot saved as {obj}_simulation_plot.png")
"""
    else:
        simulation_log_string += "\n[VISUALIZATION] Matplotlib not found. Cannot generate or preview visualization code."

    return simulation_log_string, visualization_code, can_preview_flag

def preview_visualization(visualization_code_string):
    """
    Attempts to execute the generated visualization code.
    WARNING: exec() can be dangerous with untrusted code.
    """
    if not plt:
        print("\n[PREVIEW] Matplotlib is not available. Cannot execute visualization.")
        return

    print("\n[PREVIEW] Attempting to execute visualization code...")
    print("NOTE: If the plot window appears, you may need to close it to continue.")
    try:
        exec(visualization_code_string, {'plt': plt})
        print("[PREVIEW] Visualization executed. Check for a plot window.")
    except Exception as e:
        print(f"[PREVIEW] Error during visualization execution: {e}")
        print("[PREVIEW] This could be due to GUI backend issues or errors in the generated code.")

# Main Application Logic
def main():
    """
    Main function to run the text-to-simulation application.
    """
    print("--- Simple Text-to-Simulation (with Gemini API Placeholder & Visualization) ---")
    if not plt:
        print("WARNING: Matplotlib library not found. Plot generation and preview will be disabled.")
        print("Please install it if you want visualization: pip install matplotlib")

    while True:
        user_prompt = input("\nDescribe the simulation (e.g., 'a ball moving at 10 m/s for 5 seconds') or type 'exit': \n> ")

        if user_prompt.lower() == 'exit':
            print("Exiting application.")
            break

        if not user_prompt.strip():
            print("Please enter a description.")
            continue

        simulation_params = call_gemini_api(user_prompt)

        if simulation_params:
            sim_log, viz_code, can_preview = run_simulation_and_generate_code(simulation_params)
            print(sim_log)

            if viz_code:
                print("\n--- Generated Visualization Code (Python with Matplotlib) ---")
                print(viz_code)
                print("-----------------------------------------------------------")

                if can_preview:
                    preview_choice = input("Attempt to preview the visualization? (y/n): ").lower()
                    if preview_choice == 'y':
                        preview_visualization(viz_code)
                else:
                    print("Preview unavailable (Matplotlib issue).")
            else:
                print("\n[MAIN APP] No visualization code generated.")
        else:
            print("\n[MAIN APP] Could not run simulation due to issues with parsing the prompt.")

if __name__ == "__main__":
    main()
