"""
================================================================================
BASELINE DIGITAL TWIN - RK4 VERSION (C++ READY)
================================================================================
Description:
This script acts as the discrete-time "Physics Plant Model" of the 5.6-kW
Baldor PM-SyRM. It runs at a fixed 4 kHz cycle time (250 µs) to match the
ABB Crealizer environment.

INPUTS & OUTPUTS:
- Inputs:  Commanded Voltages (Vd, Vq) and mechanical speed (RPM).
- Outputs: Estimated Stator Currents (id, iq).

* Note on Architecture vs. Gradnet (Task 2.1.2):
  This script represents the physical motor (Voltage in -> Current out).
  In contrast, the Gradient Neural Network is trained on the
  direct map: it takes the measured currents (id, iq) as INPUTS to
  estimate the internal flux linkages (psi_d, psi_q) as OUTPUTS.

MATHEMATICAL SOLVER: 4th-Order Runge-Kutta (RK4)
- Pros: Highly accurate. Matches continuous mathematical models very closely.
- Cons: Requires 4 derivative calculations (and 4 LUT lookups) per time step.
  This demands 4x more CPU power than the Forward Euler version.

NEXT STEPS:
1. C++ Translation:
   This RK4 logic can be translated into C++ for maximum accuracy. However, if
   the Crealizer hardware throws a CPU overload error during the T1 (250 µs)
   task, fall back to the simpler Forward Euler version.

2. Neural Network Integration:
   If the AI Gradnet replaces the CSV lookups, this RK4 method will
   force the AI to run 4 times every 250 µs. Ensure the neural network
   inference time is fast enough, or use the Forward Euler version instead.
================================================================================
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class BaselineDigitalTwinRK4:
    def __init__(self):
        # 1. Physical Motor Parameters (5.6-kW Baldor PM-SyRM)
        self.Rs = 0.63  # Stator resistance in Ohms
        self.n_p = 2  # Number of pole pairs
        self.dt = 1 / 4000.0  # Crealizer cycle time (4 kHz)

        # 2. Initialize State Variables (Magnetic Flux)
        self.psi_d = 0.0
        self.psi_q = 0.0

        # 3. Initialize Outputs (Current)
        self.i_d = 0.0
        self.i_q = 0.0

        # 4. Setup the 2D Lookup Table (LUT) for C++ Translation
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

    def _get_derivatives(self, psi_d_val, psi_q_val, Vd, Vq, w_e):
        """Helper function for RK4: Calculates the slopes (d_psi/dt)."""
        i_d_val, i_q_val = self.get_currents_from_flux(psi_d_val, psi_q_val)
        dpsi_d = Vd - self.Rs * i_d_val + w_e * psi_q_val
        dpsi_q = Vq - self.Rs * i_q_val - w_e * psi_d_val
        return dpsi_d, dpsi_q

    def update_step(self, Vd, Vq, rpm):
        """
        The discrete time step loop using 4th-Order Runge-Kutta (Runs at 4 kHz).
        """
        w_e = self.n_p * rpm * (2 * np.pi / 60.0)

        # k1: Slope at the beginning of the step
        kd1, kq1 = self._get_derivatives(self.psi_d, self.psi_q, Vd, Vq, w_e)

        # k2: Slope at the midpoint (using k1)
        psi_d_k2 = self.psi_d + 0.5 * self.dt * kd1
        psi_q_k2 = self.psi_q + 0.5 * self.dt * kq1
        kd2, kq2 = self._get_derivatives(psi_d_k2, psi_q_k2, Vd, Vq, w_e)

        # k3: Slope at the midpoint (using k2)
        psi_d_k3 = self.psi_d + 0.5 * self.dt * kd2
        psi_q_k3 = self.psi_q + 0.5 * self.dt * kq2
        kd3, kq3 = self._get_derivatives(psi_d_k3, psi_q_k3, Vd, Vq, w_e)

        # k4: Slope at the end of the step (using k3)
        psi_d_k4 = self.psi_d + self.dt * kd3
        psi_q_k4 = self.psi_q + self.dt * kq3
        kd4, kq4 = self._get_derivatives(psi_d_k4, psi_q_k4, Vd, Vq, w_e)

        # Combine them for the final highly-accurate state update
        self.psi_d += (self.dt / 6.0) * (kd1 + 2 * kd2 + 2 * kd3 + kd4)
        self.psi_q += (self.dt / 6.0) * (kq1 + 2 * kq2 + 2 * kq3 + kq4)

        # Update final output currents
        self.i_d, self.i_q = self.get_currents_from_flux(self.psi_d, self.psi_q)
        return self.i_d, self.i_q


# --- Test the logic ---
if __name__ == "__main__":
    twin = BaselineDigitalTwinRK4()
    print("Testing discrete Digital Twin RK4 step...")
    id_est, iq_est = twin.update_step(Vd=10.0, Vq=200.0, rpm=1500)
    print(f"Output Current: id = {id_est:.2f} A, iq = {iq_est:.2f} A")
