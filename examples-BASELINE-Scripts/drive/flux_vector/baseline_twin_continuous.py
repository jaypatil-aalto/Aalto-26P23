"""
================================================================================
CONTINUOUS BASELINE DIGITAL TWIN (MOTULATOR EXACT MATCH)
================================================================================
Description:
This script acts as the "Perfect Math" Continuous Plant Model. It uses SciPy's
variable-step ODE solvers (RK45) to solve the motor's differential equations
with microscopic accuracy, matching the original Motulator example perfectly.

This script requires continuous, variable-sized time steps. The ABB
Crealizer operates on a strict, fixed 250 µs multi-tasking cycle (T1) (or other fixed cycles).
Trying to implement this variable-step solver in C++ will likely cause CPU
overloads and system crashes. Use the cpp ready models instead.
================================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator


class ContinuousBaselineTwin:
    def __init__(self):
        # 1. Physical Motor Parameters (5.6-kW Baldor PM-SyRM)
        self.Rs = 0.63  # Stator resistance in Ohms
        self.n_p = 2  # Number of pole pairs

        # 2. Setup the 2D Lookup Table (LUT)
        self._setup_flux_to_current_lut()

    def _setup_flux_to_current_lut(self):
        """Loads the real Baldor motor saturation map from CSV files."""
        psi_d_grid = np.loadtxt("lut_psi_d_axis.csv", delimiter=",")
        psi_q_grid = np.loadtxt("lut_psi_q_axis.csv", delimiter=",")
        id_table = np.loadtxt("lut_id_table.csv", delimiter=",")
        iq_table = np.loadtxt("lut_iq_table.csv", delimiter=",")

        self.lut_id = RegularGridInterpolator(
            (psi_d_grid, psi_q_grid), id_table, bounds_error=False, fill_value=None
        )
        self.lut_iq = RegularGridInterpolator(
            (psi_d_grid, psi_q_grid), iq_table, bounds_error=False, fill_value=None
        )

    def get_currents_from_flux(self, psi_d, psi_q):
        """Reads the 2D Lookup Table."""
        id_est = self.lut_id((psi_d, psi_q))
        iq_est = self.lut_iq((psi_d, psi_q))
        return float(id_est), float(iq_est)

    def motor_dynamics(self, t, states, Vd, Vq, rpm):
        """
        The continuous differential equations (ODE).
        Instead of taking a fixed step, this function calculates the exact
        slopes (derivatives) so the SciPy solver can integrate them perfectly.
        """
        psi_d, psi_q = states

        # 1. Use the Saturation LUT to find the currents
        i_d, i_q = self.get_currents_from_flux(psi_d, psi_q)

        # 2. Convert mechanical RPM to electrical angular velocity (rad/s)
        w_e = self.n_p * rpm * (2 * np.pi / 60.0)

        # 3. Calculate the continuous derivatives (d_psi / dt)
        dpsi_d_dt = Vd - self.Rs * i_d + w_e * psi_q
        dpsi_q_dt = Vq - self.Rs * i_q - w_e * psi_d

        return [dpsi_d_dt, dpsi_q_dt]


# --- Testing the continuous model functionality by plotting ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    twin = ContinuousBaselineTwin()

    # 1. Simulation parameters
    t_span = (0.0, 0.05)  # Simulate from 0 to 50 milliseconds
    initial_states = [0.0, 0.0]  # Initial flux [psi_d, psi_q]

    # Test Inputs: Small step in voltage
    Vd_test = 2.0
    Vq_test = 10.0
    rpm_test = 0.0

    print("Running continuous variable-step ODE solver (RK45)...")

    # 2. Run the SciPy Continuous Solver
    solution = solve_ivp(
        fun=twin.motor_dynamics,
        t_span=t_span,
        y0=initial_states,
        args=(Vd_test, Vq_test, rpm_test),
        dense_output=True,
        max_step=0.001,  # Forces the solver to give us enough points for a smooth plot
    )

    # 3. Extract the data for plotting
    t_plot = np.linspace(0, 0.05, 500)
    psi_states = solution.sol(t_plot)

    id_data = []
    iq_data = []

    # Convert the solved flux states back into currents for the graph
    for i in range(len(t_plot)):
        id_est, iq_est = twin.get_currents_from_flux(psi_states[0, i], psi_states[1, i])
        id_data.append(id_est)
        iq_data.append(iq_est)

    # 4. The Plot
    plt.figure(figsize=(10, 5))
    plt.plot(t_plot, id_data, label="i_d (d-axis current)", linewidth=2)
    plt.plot(t_plot, iq_data, label="i_q (q-axis current)", linewidth=2)
    plt.title("Continuous Baseline Model (Exact Motulator Match)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Current (Amperes)")
    plt.grid(True)
    plt.legend()

    print("Close the plot window to finish the script.")
    plt.show()
