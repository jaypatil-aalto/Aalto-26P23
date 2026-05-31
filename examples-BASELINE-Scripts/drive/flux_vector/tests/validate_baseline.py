import os
import sys
import traceback

# =====================================================================
# Tell Python to look in the parent folder (flux_vector) for
# the Python models, but DO NOT change the working directory,
# because the CSV files are sitting in the workspace root!
# =====================================================================
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
import numpy as np

print("\n" + "=" * 50)
print(" STARTING VALIDATION SCRIPT")
print("=" * 50)
print(
    "Running Motulator simulation... (Please CLOSE the Motulator graphs when they appear to continue!)"
)

# Catch import errors just in case
try:
    import plot_6kw_pmsyrm_sat_fvc as mot_sim
    from baseline_twin_cpp_ready import BaselineDigitalTwin
    from baseline_twin_rk4_cpp_ready import BaselineDigitalTwinRK4
except Exception:
    print("\n CRASH DURING IMPORTS:")
    traceback.print_exc()
    sys.exit()


def run_validation():
    try:
        print("\n1. Extracting continuous-time ground truth data from Motulator...")
        res = mot_sim.res

        # Grab raw continuous data
        t_cont = np.array(res.mdl.t)
        w_M_cont = np.array(res.mdl.mechanics.w_M)
        i_s_dq = np.array(res.mdl.machine.i_s_dq)

        # Grab stationary voltage (alpha-beta) and electrical angle
        u_s_ab = np.array(res.mdl.machine.u_s_ab)
        theta_m = np.array(res.mdl.machine.theta_m)

        # Park Transformation: Rotate voltage to synchronous dq frame
        u_s_dq = u_s_ab * np.exp(-1j * theta_m)

        # Extract the D and Q axis components
        vd_cont = u_s_dq.real
        vq_cont = u_s_dq.imag
        id_ground_truth = i_s_dq.real
        iq_ground_truth = i_s_dq.imag

        # Convert mechanical rad/s to RPM for the Crealizer input
        rpm_cont = w_M_cont * (60 / (2 * np.pi))

        print("2. Interpolating inputs to fixed 4 kHz Crealizer cycle time...")
        dt = 1.0 / 4000.0
        t_discrete = np.arange(0, 2.0, dt)

        vd_discrete = np.interp(t_discrete, t_cont, vd_cont)
        vq_discrete = np.interp(t_discrete, t_cont, vq_cont)
        rpm_discrete = np.interp(t_discrete, t_cont, rpm_cont)

        id_truth_discrete = np.interp(t_discrete, t_cont, id_ground_truth)
        iq_truth_discrete = np.interp(t_discrete, t_cont, iq_ground_truth)

        print("3. Feeding inputs step-by-step into both discrete Baseline Twins...")
        euler_twin = BaselineDigitalTwin()
        rk4_twin = BaselineDigitalTwinRK4()

        id_euler, iq_euler = np.zeros_like(t_discrete), np.zeros_like(t_discrete)
        id_rk4, iq_rk4 = np.zeros_like(t_discrete), np.zeros_like(t_discrete)

        for k in range(len(t_discrete)):
            # Update Forward Euler
            id_e, iq_e = euler_twin.update_step(
                vd_discrete[k], vq_discrete[k], rpm_discrete[k]
            )
            id_euler[k], iq_euler[k] = id_e, iq_e

            # Update RK4
            id_r, iq_r = rk4_twin.update_step(
                vd_discrete[k], vq_discrete[k], rpm_discrete[k]
            )
            id_rk4[k], iq_rk4[k] = id_r, iq_r

        print("4. Calculating Cumulative Errors (Mean Absolute Error)...")
        mae_id_euler = np.mean(np.abs(id_truth_discrete - id_euler))
        mae_iq_euler = np.mean(np.abs(iq_truth_discrete - iq_euler))

        mae_id_rk4 = np.mean(np.abs(id_truth_discrete - id_rk4))
        mae_iq_rk4 = np.mean(np.abs(iq_truth_discrete - iq_rk4))

        print("-" * 50)
        print(
            f"Forward Euler Average Error -> Id: {mae_id_euler:.4f} A | Iq: {mae_iq_euler:.4f} A"
        )
        print(
            f"RK4 Average Error         -> Id: {mae_id_rk4:.4f} A | Iq: {mae_iq_rk4:.4f} A"
        )
        print("-" * 50)

        # =====================================================================
        # 5. Plot the Model Inputs (Voltages and Speed)
        # =====================================================================
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(
            t_discrete,
            vd_discrete,
            label="Vd (D-Axis Voltage)",
            color="blue",
            alpha=0.8,
        )
        plt.plot(
            t_discrete, vq_discrete, label="Vq (Q-Axis Voltage)", color="red", alpha=0.8
        )
        plt.title("Model Inputs: Stator Voltages (Synchronous dq-frame)")
        plt.ylabel("Voltage (V)")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(
            t_discrete,
            rpm_discrete,
            label="Mechanical Speed (RPM)",
            color="green",
            alpha=0.8,
        )
        plt.title("Model Inputs: Motor Speed")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (RPM)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # =====================================================================
        # 6. Plot the Output Differences (Currents)
        # =====================================================================
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(
            t_discrete,
            id_truth_discrete,
            label="Motulator (Ground Truth)",
            color="black",
            linestyle="--",
        )
        plt.plot(t_discrete, id_euler, label="Forward Euler Twin", alpha=0.7)
        plt.plot(t_discrete, id_rk4, label="RK4 Twin", alpha=0.7)
        plt.title("D-Axis Current (Id) Estimation vs Ground Truth")
        plt.ylabel("Current (A)")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(
            t_discrete,
            iq_truth_discrete,
            label="Motulator (Ground Truth)",
            color="black",
            linestyle="--",
        )
        plt.plot(t_discrete, iq_euler, label="Forward Euler Twin", alpha=0.7)
        plt.plot(t_discrete, iq_rk4, label="RK4 Twin", alpha=0.7)
        plt.title("Q-Axis Current (Iq) Estimation vs Ground Truth")
        plt.xlabel("Time (s)")
        plt.ylabel("Current (A)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    except Exception:
        print("\n CRASH DURING VALIDATION MATH:")
        traceback.print_exc()
        input("Press Enter to exit so the terminal stays open...")


if __name__ == "__main__":
    run_validation()
