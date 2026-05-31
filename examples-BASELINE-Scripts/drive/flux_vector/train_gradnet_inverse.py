import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


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
        # Note: mathematically this now takes flux (psi_s) as input!
        return 0.5 * (self._g(i_s) + self._g(i_s @ self.C) @ self.C)


# =====================================================================
# DATA SECTION
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "baldor_400rpm_map.npz")

print(f"Looking for data at: {data_path}")
data = np.load(data_path)

# SWAPPED: X is now Flux (Input), Y is now Current (Output)
X_np = np.column_stack(
    (data["psi_s_dq"].flatten().real, data["psi_s_dq"].flatten().imag)
).astype(np.float32)
Y_np = np.column_stack(
    (data["i_s_dq"].flatten().real, data["i_s_dq"].flatten().imag)
).astype(np.float32)

Psi_base = np.abs(X_np).max()  # X is Flux
I_base = np.abs(Y_np).max()  # Y is Current

X_t = torch.tensor(X_np / Psi_base)
Y_t = torch.tensor(Y_np / I_base)
loader = DataLoader(TensorDataset(X_t, Y_t), batch_size=32, shuffle=True)

# =====================================================================
# TRAINING SECTION
# =====================================================================
model = GradNetFluxMap(N=12)
opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sch = optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)

print("Training Inverse Model (Flux -> Current)...")
print(
    f"Params: {sum(p.numel() for p in model.parameters())} | {len(X_np)} points | Psi_base={Psi_base:.4f} Vs | I_base={I_base:.1f} A"
)

for ep in range(5000):
    ep_loss = 0.0
    for xb, yb in loader:
        opt.zero_grad()
        loss = nn.MSELoss()(model(xb), yb)
        loss.backward()
        opt.step()
        ep_loss += loss.item()
    sch.step()
    if (ep + 1) % 500 == 0:
        print(
            f"Epoch {ep + 1:5d} | Loss: {ep_loss / len(loader):.4e} | LR: {sch.get_last_lr()[0]:.2e}"
        )

# =====================================================================
# VALIDATION SECTION
# =====================================================================
model.eval()
with torch.no_grad():
    errors = np.linalg.norm(Y_t.numpy() - model(X_t).numpy(), axis=1)

e_rms, e_max = np.sqrt(np.mean(errors**2)), np.max(errors)
print(
    f"\ne_rms={e_rms:.4f} p.u. ({e_rms * I_base:.3f} A) | e_max={e_max:.4f} p.u. ({e_max * I_base:.3f} A)"
)
print(
    f"beta={torch.exp(model.activation.log_beta).item():.4f} | "
    f"mu_d={torch.exp(model.log_mu[0]).item():.4f} | mu_q={torch.exp(model.log_mu[1]).item():.4f}"
)

# Inference time (Using dummy flux values instead of currents)
x_test = torch.tensor([[0.5 / Psi_base, 0.8 / Psi_base]], dtype=torch.float32)
with torch.no_grad():
    for _ in range(500):
        model(x_test)
    times = []
    for _ in range(10000):
        t0 = time.perf_counter_ns()
        model(x_test)
        times.append(time.perf_counter_ns() - t0)
print(
    f"Inference: median={np.median(times) / 1000:.0f} us | min={np.min(times) / 1000:.0f} us"
)

# =====================================================================
# SAVE SECTION
# =====================================================================
# Save the model in the exact same directory as the script
save_path = os.path.join(current_dir, "baldor_gradnet_inverse.pth")
torch.save(
    {
        "model_state": model.state_dict(),
        "I_base": I_base,
        "Psi_base": Psi_base,
        "N": 12,
        "e_rms": float(e_rms),
        "e_max": float(e_max),
    },
    save_path,
)
print(f"Saved: {save_path}")
