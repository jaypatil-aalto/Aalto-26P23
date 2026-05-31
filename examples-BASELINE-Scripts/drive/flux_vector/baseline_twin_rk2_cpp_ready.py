"""
================================================================================
BASELINE DIGITAL TWIN - RK2 VERSION (HEUN's METHOD / C++ READY)
================================================================================
Description:
This script acts as the discrete-time "Physics Plant Model" of the 5.6-kW
Baldor PM-SyRM. It runs at a fixed 4 kHz cycle time (250 µs) to match the
ABB Crealizer environment.

INPUTS & OUTPUTS:
- Inputs:  Commanded Voltages (Vd, Vq) and mechanical speed (RPM).
- Outputs: Estimated Stator Currents (id, iq).

MATHEMATICAL SOLVER: 2nd-Order Runge-Kutta (RK2 / Heun's Method)
- Pros: Twice as fast as RK4 (requires only 2 derivative calculations per step).
  Significantly more stable than Forward Euler during transient voltage spikes.
- Cons: Slightly less accurate than RK4 for highly non-linear dynamics, but
  provides an excellent "middle-ground" for real-time embedded control.
================================================================================
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class BaselineDigitalTwinRK2:
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
        """Helper function for RK2: Calculates the slopes (d_psi/dt)."""
        i_d_val, i_q_val = self.get_currents_from_flux(psi_d_val, psi_q_val)
        dpsi_d = Vd - self.Rs * i_d_val + w_e * psi_q_val
        dpsi_q = Vq - self.Rs * i_q_val - w_e * psi_d_val
        return dpsi_d, dpsi_q

    def update_step(self, Vd, Vq, rpm):
        """
        The discrete time step loop using 2nd-Order Runge-Kutta (Runs at 4 kHz).
        """
        w_e = self.n_p * rpm * (2 * np.pi / 60.0)

        # k1: Slope at the beginning of the step
        kd1, kq1 = self._get_derivatives(self.psi_d, self.psi_q, Vd, Vq, w_e)

        # k2: Slope at the FULL future step (using k1 prediction)
        psi_d_k2 = self.psi_d + self.dt * kd1
        psi_q_k2 = self.psi_q + self.dt * kq1
        kd2, kq2 = self._get_derivatives(psi_d_k2, psi_q_k2, Vd, Vq, w_e)

        # Combine them by taking the average
        self.psi_d += (self.dt / 2.0) * (kd1 + kd2)
        self.psi_q += (self.dt / 2.0) * (kq1 + kq2)

        # Update final output currents
        self.i_d, self.i_q = self.get_currents_from_flux(self.psi_d, self.psi_q)
        return self.i_d, self.i_q


# --- Test the logic ---
if __name__ == "__main__":
    twin = BaselineDigitalTwinRK2()
    print("Testing discrete Digital Twin RK2 step...")
    id_est, iq_est = twin.update_step(Vd=10.0, Vq=200.0, rpm=1500)
    print(f"Output Current: id = {id_est:.2f} A, iq = {iq_est:.2f} A")
