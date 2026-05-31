"""
================================================================================
BASELINE DIGITAL TWIN (C++ READY)
================================================================================
Description:
This script acts as the discrete-time "Physics Plant Model" of the 5.6-kW
Baldor PM-SyRM. It simulates the bare-metal electrical physics of the motor
running at a fixed 4 kHz cycle time to match the ABB Crealizer environment.
(Forward Euler version of the baseline code)

INPUTS & OUTPUTS:
- Inputs:  Commanded Voltages (Vd, Vq) and mechanical speed (RPM).
- Outputs: Estimated Stator Currents (id, iq).

* Note on Architecture vs. Gradnet (Task 2.1.2):
  This script represents the physical motor (Voltage in -> Current out).
  In contrast, Jay's Gradient Neural Network (Gradnet) is trained on the
  direct map: it takes the measured currents (id, iq) as INPUTS to
  estimate the internal flux linkages (psi_d, psi_q) as OUTPUTS.

NEXT STEPS:
1. C++ Translation:
   This script is used to build the C++ class, loading the
   attached CSV files into 2D arrays to replicate the lookup table (LUT) logic.

2. Neural Network Integration:
   Currently, the non-linear magnetic saturation is handled by the CSV LUTs.
   Once the "Gradnet" is validated, it will completely replace the
   `get_currents_from_flux()` function, calculating the physics on the fly
   and eliminating the need for the CSVs.
================================================================================
"""

# import matplotlib.pyplot as plt  # needed just for the plots
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class BaselineDigitalTwin:
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
        """
        Loads the real Baldor motor saturation map from CSV files.
        In C++, Katarzyna will load these exact same CSVs into standard 2D arrays.
        """
        # Load the 1D axes
        psi_d_grid = np.loadtxt("lut_psi_d_axis.csv", delimiter=",")
        psi_q_grid = np.loadtxt("lut_psi_q_axis.csv", delimiter=",")

        # Load the 2D current tables
        id_table = np.loadtxt("lut_id_table.csv", delimiter=",")
        iq_table = np.loadtxt("lut_iq_table.csv", delimiter=",")

        # Create the interpolator using the real data
        self.lut_id = RegularGridInterpolator(
            (psi_d_grid, psi_q_grid), id_table, bounds_error=False, fill_value=None
        )
        self.lut_iq = RegularGridInterpolator(
            (psi_d_grid, psi_q_grid), iq_table, bounds_error=False, fill_value=None
        )

    def get_currents_from_flux(self, psi_d, psi_q):
        """
        Reads the 2D Lookup Table. In C++, this will be a simple bilinear interpolation function.
        """
        id_est = self.lut_id((psi_d, psi_q))
        iq_est = self.lut_iq((psi_d, psi_q))
        return float(id_est), float(iq_est)

    def update_step(self, Vd, Vq, rpm):
        """
        The discrete time step loop (Runs at 4 kHz).
        """
        # 1. Convert mechanical RPM to electrical angular velocity (rad/s)
        w_m = rpm * (2 * np.pi / 60.0)
        w_e = self.n_p * w_m

        # 2. Update Flux States using Forward Euler Integration
        psi_d_next = self.psi_d + self.dt * (Vd - self.Rs * self.i_d + w_e * self.psi_q)
        psi_q_next = self.psi_q + self.dt * (Vq - self.Rs * self.i_q - w_e * self.psi_d)

        # 3. Save new states
        self.psi_d = psi_d_next
        self.psi_q = psi_q_next

        # 4. Use the Saturation LUT to find the new currents
        self.i_d, self.i_q = self.get_currents_from_flux(self.psi_d, self.psi_q)

        # 5. Output the estimated currents
        return self.i_d, self.i_q


# --- Test the logic ---
if __name__ == "__main__":
    twin = BaselineDigitalTwin()
    print("Testing discrete Digital Twin step...")
    id_est, iq_est = twin.update_step(Vd=10.0, Vq=200.0, rpm=1500)
    print(f"Output Current: id = {id_est:.2f} A, iq = {iq_est:.2f} A")


"""
# Testing the baseline model functionality by plotting the results
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    twin = BaselineDigitalTwin()

    # 1. Setup storage arrays for the plot
    time_data = []
    id_data = []
    iq_data = []

    # 2. Simulation parameters (Simulate 50 milliseconds)
    t_final = 0.05
    num_steps = int(t_final / twin.dt)

    print(f"Simulating {num_steps} discrete steps for plotting...")

    # 3. Run the 4 kHz loop over time
    for step in range(num_steps):
        t = step * twin.dt

        # Test Inputs: Sudden step in voltage
        Vd_test = 2.0
        Vq_test = 10.0
        rpm_test = 0.0

        # Step the twin forward by one dt
        id_est, iq_est = twin.update_step(Vd_test, Vq_test, rpm_test)

        # Save the data points
        time_data.append(t)
        id_data.append(id_est)
        iq_data.append(iq_est)

    # 4. The Plot (You can easily comment this block out later!)
    plt.figure(figsize=(10, 5))
    plt.plot(time_data, id_data, label="i_d (d-axis current)", linewidth=2)
    plt.plot(time_data, iq_data, label="i_q (q-axis current)", linewidth=2)
    plt.title("Discrete Baseline Model: Current Step Response")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Current (Amperes)")
    plt.grid(True)
    plt.legend()

    print("Close the plot window to finish the script.")
    plt.show()
"""
