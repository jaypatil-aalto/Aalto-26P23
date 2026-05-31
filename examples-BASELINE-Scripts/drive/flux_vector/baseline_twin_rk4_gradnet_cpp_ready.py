"""
================================================================================
BASELINE DIGITAL TWIN - RK4 + GRADNET VERSION
================================================================================
Description:
This script acts as the discrete-time "Physics Plant Model" of the 5.6-kW
Baldor PM-SyRM. It runs at a fixed 4 kHz cycle time (250 µs).

Instead of using CSV Lookup Tables (LUTs) for the flux-to-current mapping,
this version uses a trained PyTorch Neural Network (GradNet) to estimate
non-linear saturation dynamically.

REQUIREMENT:
You must have the 'baldor_gradnet_inverse.pth' file in the same directory,
trained to take Flux as INPUT and predict Current as OUTPUT.
================================================================================
"""

import os

import numpy as np
import torch
from torch import nn


# =====================================================================
# PYTORCH MODEL DEFINITIONS (Must match the training script exactly)
# =====================================================================
class AlgebraicSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_beta = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        beta = torch.exp(self.log_beta)
        return z / torch.sqrt(z**2 + beta)


class GradNetFluxMap(nn.Module):
    def __init__(self, N=12, mu_init=0.01):
        super().__init__()
        self.A = nn.Parameter(torch.randn(N, 2) * 0.1)
        self.b = nn.Parameter(torch.zeros(N))
        self.b0 = nn.Parameter(torch.zeros(2))
        self.log_mu = nn.Parameter(torch.full((2,), np.log(mu_init)))
        self.activation = AlgebraicSigmoid()
        self.register_buffer("C", torch.diag(torch.tensor([1.0, -1.0])))

    def _g(self, x):
        mu = torch.exp(self.log_mu)
        z = x @ self.A.T + self.b
        return x @ torch.diag(mu) + self.b0 + self.activation(z) @ self.A

    def forward(self, i_s):
        return 0.5 * (self._g(i_s) + self._g(i_s @ self.C) @ self.C)


# =====================================================================
# THE DIGITAL TWIN PHYSICS ENGINE
# =====================================================================
class BaselineDigitalTwinRK4GradNet:
    def __init__(self, model_filename="baldor_gradnet_inverse.pth"):
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

        # 4. Load the PyTorch Neural Network
        self._load_ai_model(model_filename)

    def _load_ai_model(self, model_filename):
        """Loads the trained PyTorch model and scaling bases using robust paths."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Could not find the Neural Network model: {model_path}"
            )

        # weights_only=True is safer and the new PyTorch standard
        checkpoint = torch.load(model_path, weights_only=False)

        # Initialize the blank model architecture
        self.ai_model = GradNetFluxMap(N=checkpoint["N"])

        # Load the trained weights
        self.ai_model.load_state_dict(checkpoint["model_state"])

        # CRITICAL: Set to evaluation mode to disable gradients/training behavior
        self.ai_model.eval()

        # Load scaling bases
        self.I_base = checkpoint["I_base"]
        self.Psi_base = checkpoint["Psi_base"]

    def get_currents_from_flux(self, psi_d, psi_q):
        """Uses the Neural Network to map Flux to Current."""
        # 1. Normalize the inputs using the Psi_base
        psi_tensor = torch.tensor(
            [[psi_d / self.Psi_base, psi_q / self.Psi_base]], dtype=torch.float32
        )

        # 2. Run AI inference
        # torch.no_grad() is absolutely mandatory here so it doesn't leak memory!
        with torch.no_grad():
            i_pred_tensor = self.ai_model(psi_tensor)

        # 3. De-normalize the outputs back to real Amperes
        id_est = i_pred_tensor[0, 0].item() * self.I_base
        iq_est = i_pred_tensor[0, 1].item() * self.I_base

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
    try:
        twin = BaselineDigitalTwinRK4GradNet()
        print("Testing discrete Digital Twin RK4 + AI inference step...")
        id_est, iq_est = twin.update_step(Vd=10.0, Vq=200.0, rpm=1500)
        print(f"Output Current: id = {id_est:.2f} A, iq = {iq_est:.2f} A")
        print("Success! The Neural Network successfully replaced the CSV tables.")
    except Exception as e:
        print(f"Failed to run test.\nError: {e}")
