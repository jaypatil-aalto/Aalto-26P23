import os
import sys
import time
import traceback

# =====================================================================
# Tell Python to look in the parent folder (flux_vector) for
# the Python models, but DO NOT change the working directory,
# because the CSV and .pth files are sitting in the workspace root!
# =====================================================================
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
import numpy as np

print("\n" + "=" * 60)
print("STARTING RK4 LUT vs GRADNET VALIDATION SCRIPT")
print("=" * 60)
print(
    "Running Motulator simulation... (Please CLOSE the Motulator graphs when they appear to continue!)"
)

# Catch import errors just in case
try:
    import plot_6kw_pmsyrm_sat_fvc as mot_sim

    # Import BOTH Digital Twins
    from baseline_twin_rk4_cpp_ready import BaselineDigitalTwinRK4
    from baseline_twin_rk4_gradnet_cpp_ready import BaselineDigitalTwinRK4GradNet
except Exception:
    print("\nCRASH DURING IMPORTS:")
    traceback.print_exc()
    sys.exit()


def run_validation():
    try:
        print("\n1. Extracting continuous-time ground truth data from Motulator...")
        res = mot_sim.res

        # =====================================================================
        # DATA EXTRACTION
        # =====================================================================
        t_cont = np.array(res.mdl.t)
        w_M_cont = np.array(res.mdl.mechanics.w_M)

        i_s_dq = np.array(res.mdl.machine.i_s_dq)
        u_s_ab = np.array(res.mdl.machine.u_s_ab)
        theta_m = np.array(res.mdl.machine.theta_m)

        u_s_dq = u_s_ab * np.exp(-1j * theta_m)

        vd_cont = u_s_dq.real
        vq_cont = u_s_dq.imag
        id_ground_truth = i_s_dq.real
        iq_ground_truth = i_s_dq.imag

        rpm_cont = w_M_cont * (60 / (2 * np.pi))

        print("2. Interpolating inputs to fixed 4 kHz Crealizer cycle time...")
        dt = 1.0 / 4000.0
        t_discrete = np.arange(0, 2.0, dt)

        vd_discrete = np.interp(t_discrete, t_cont, vd_cont)
        vq_discrete = np.interp(t_discrete, t_cont, vq_cont)
        rpm_discrete = np.interp(t_discrete, t_cont, rpm_cont)

        id_truth_discrete = np.interp(t_discrete, t_cont, id_ground_truth)
        iq_truth_discrete = np.interp(t_discrete, t_cont, iq_ground_truth)

        print(
            "3. Feeding inputs step-by-step into the RK4 LUT and RK4 GradNet Twins..."
        )
        print(
            "   (Note: PyTorch inference in a pure Python loop takes time. Please wait...)"
        )

        rk4_lut_twin = BaselineDigitalTwinRK4()
        rk4_ai_twin = BaselineDigitalTwinRK4GradNet()

        id_rk4_lut, iq_rk4_lut = np.zeros_like(t_discrete), np.zeros_like(t_discrete)
        id_rk4_ai, iq_rk4_ai = np.zeros_like(t_discrete), np.zeros_like(t_discrete)

        start_time = time.time()

        for k in range(len(t_discrete)):
            # Simulate one step for RK4 with CSV LUTs
            id_lut, iq_lut = rk4_lut_twin.update_step(
                vd_discrete[k], vq_discrete[k], rpm_discrete[k]
            )
            id_rk4_lut[k], iq_rk4_lut[k] = id_lut, iq_lut

            # Simulate one step for RK4 with PyTorch GradNet
            id_ai, iq_ai = rk4_ai_twin.update_step(
                vd_discrete[k], vq_discrete[k], rpm_discrete[k]
            )
            id_rk4_ai[k], iq_rk4_ai[k] = id_ai, iq_ai

        sim_time = time.time() - start_time
        print(f"   Done! AI Simulation Loop took {sim_time:.1f} seconds.")

        print("4. Calculating Cumulative Errors (Mean Absolute Error)...")
        mae_id_lut = np.mean(np.abs(id_truth_discrete - id_rk4_lut))
        mae_iq_lut = np.mean(np.abs(iq_truth_discrete - iq_rk4_lut))

        mae_id_ai = np.mean(np.abs(id_truth_discrete - id_rk4_ai))
        mae_iq_ai = np.mean(np.abs(iq_truth_discrete - iq_rk4_ai))

        print("-" * 60)
        print(
            f"RK4 LUT Average Error       -> Id: {mae_id_lut:.4f} A | Iq: {mae_iq_lut:.4f} A"
        )
        print(
            f"RK4 GradNet Average Error   -> Id: {mae_id_ai:.4f} A | Iq: {mae_iq_ai:.4f} A"
        )
        print("-" * 60)

        # =====================================================================
        # 5. Plot the Output Differences (Currents)
        # =====================================================================
        plt.figure(figsize=(12, 8))

        # Plot D-Axis Currents
        plt.subplot(2, 1, 1)
        plt.plot(
            t_discrete,
            id_truth_discrete,
            label="Motulator (Ground Truth)",
            color="black",
            linestyle="--",
            linewidth=2,
        )
        plt.plot(t_discrete, id_rk4_lut, label="RK4 LUT Twin", color="blue", alpha=0.8)
        plt.plot(
            t_discrete,
            id_rk4_ai,
            label="RK4 GradNet Twin",
            color="red",
            alpha=0.8,
            linestyle=":",
        )
        plt.title("D-Axis Current (Id) Estimation: LUT vs AI")
        plt.ylabel("Current (A)")
        plt.legend()
        plt.grid(True)

        # Plot Q-Axis Currents
        plt.subplot(2, 1, 2)
        plt.plot(
            t_discrete,
            iq_truth_discrete,
            label="Motulator (Ground Truth)",
            color="black",
            linestyle="--",
            linewidth=2,
        )
        plt.plot(t_discrete, iq_rk4_lut, label="RK4 LUT Twin", color="blue", alpha=0.8)
        plt.plot(
            t_discrete,
            iq_rk4_ai,
            label="RK4 GradNet Twin",
            color="red",
            alpha=0.8,
            linestyle=":",
        )
        plt.title("Q-Axis Current (Iq) Estimation: LUT vs AI")
        plt.xlabel("Time (s)")
        plt.ylabel("Current (A)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    except Exception:
        print("\nCRASH DURING VALIDATION MATH:")
        traceback.print_exc()
        input("Press Enter to exit so the terminal stays open...")


if __name__ == "__main__":
    run_validation()
