import os
import sys

# Tell Python to look in the parent folder (flux_vector) for the models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import matplotlib.pyplot as plt
import numpy as np

# --- 1. IMPORT YOUR MODELS ---
from baseline_twin_cpp_ready import BaselineDigitalTwin as MotorModelEuler
from baseline_twin_rk4_cpp_ready import BaselineDigitalTwinRK4 as MotorModelRK4


def run_comparison():
    # --- 2. SETUP THE SIMULATION ---
    dt = 0.00025  # 250 us (Crealizer T1 cycle)
    t_end = 0.8  # Run for 800 milliseconds
    time = np.arange(0, t_end, dt)
    n_steps = len(time)

    # Initialize the models
    euler_model = MotorModelEuler()
    rk4_model = MotorModelRK4()

    # Arrays to store the output currents
    id_euler = np.zeros(n_steps)
    iq_euler = np.zeros(n_steps)
    id_rk4 = np.zeros(n_steps)
    iq_rk4 = np.zeros(n_steps)

    # --- 3. DEFINE A TEST SCENARIO ---
    # Let's hit the motor with a sudden voltage step while spinning
    rpm_input = 1500.0
    vd_input = 0.0
    vq_input = 100.0  # Step Q-axis voltage to 100V

    # --- 4. RUN THE LOOP ---
    for i in range(n_steps):
        # Update both models with the exact same inputs
        id_euler[i], iq_euler[i] = euler_model.update_step(
            vd_input, vq_input, rpm_input
        )
        id_rk4[i], iq_rk4[i] = rk4_model.update_step(vd_input, vq_input, rpm_input)

    # --- 5. CALCULATE CUMULATIVE ERRORS ---
    # We treat RK4 as the "Ground Truth" since it is vastly more accurate
    error_id = np.abs(id_rk4 - id_euler)
    error_iq = np.abs(iq_rk4 - iq_euler)

    # Cumulative error (sum of absolute errors over time)
    cumul_err_id = np.sum(error_id) * dt
    cumul_err_iq = np.sum(error_iq) * dt

    print("=== CUMULATIVE ERROR REPORT ===")
    print(f"Time Step: {dt * 1000000} microseconds")
    print(f"Total D-Axis Cumulative Error (A*s): {cumul_err_id:.6f}")
    print(f"Total Q-Axis Cumulative Error (A*s): {cumul_err_iq:.6f}")
    print(f"Max Instantaneous Error (Q-Axis): {np.max(error_iq):.4f} Amps")

    # --- 6. PLOT THE RESULTS ---
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(
        time, iq_rk4, label="RK4 (Ground Truth)", color="black", linestyle="dashed"
    )
    plt.plot(time, iq_euler, label="Forward Euler", color="blue", alpha=0.7)
    plt.title("Q-Axis Current ($i_q$) Response to 100V Step")
    plt.ylabel("Current (A)")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time, error_iq, label="Absolute Error |RK4 - Euler|", color="red")
    plt.title("Instantaneous Error over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (A)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_comparison()
