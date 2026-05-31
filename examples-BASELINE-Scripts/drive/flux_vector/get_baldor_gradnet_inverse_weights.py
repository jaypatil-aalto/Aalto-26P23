import os

import numpy as np
import torch

# 1. Build the path to the .pth file in the SAME directory as this script
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "baldor_gradnet_inverse.pth")

print(f"Looking for model at: {model_path}\n")

try:
    # 2. Load the weights
    ckpt = torch.load(model_path, weights_only=False)
    state = ckpt["model_state"]

    print("--- COPY THESE INTO MATLAB ---")
    print(f"I_base = {ckpt['I_base']};")
    print(f"Psi_base = {ckpt['Psi_base']};")

    # Formatting matrices for MATLAB
    A = (
        np.array2string(state["A"].numpy(), separator=", ")
        .replace("[", "")
        .replace("]", "")
        .replace("\n ", ";\n    ")
    )
    print(f"A = [{A}];")

    b = (
        np.array2string(state["b"].numpy(), separator=", ")
        .replace("[", "")
        .replace("]", "")
    )
    print(f"b = [{b}];")

    b0 = (
        np.array2string(state["b0"].numpy(), separator=", ")
        .replace("[", "")
        .replace("]", "")
    )
    print(f"b0 = [{b0}];")

    mu = (
        np.array2string(np.exp(state["log_mu"].numpy()), separator=", ")
        .replace("[", "")
        .replace("]", "")
    )
    print(f"mu = [{mu}];")

    beta = np.exp(state["activation.log_beta"].numpy()[0])
    print(f"beta = {beta};")
    print("------------------------------")

except FileNotFoundError:
    print(f"ERROR: Still cannot find the file at {model_path}")
    print("Please verify the filename matches exactly.")
