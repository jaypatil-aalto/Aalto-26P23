ABB-CREALIZER/
├── example-application/
│   ├── build/              # Generated binaries (.dll for VD+)
│   │   ├── c++/
│   │   │   ├── onnx/       # AI Research & Export artifacts
│   │   │   │   ├── baldor_gradnet_inverse.pth  # Trained PyTorch weights
│   │   │   │   ├── baldor_gradnet.onnx         # Universal AI model format
│   │   │   │   └── python_onnx.py              # Export & scaling wrapper script
│   │   │   ├── application.cpp      # Main entry point & ABB parameter I/O
│   │   │   ├── GradNetTwinRK4onnx.h # Physics engine (RK4) using ONNX model
│   │   │   ├── model_generated.c    # Compiled AI math (from onnx2c)
│   │   │   ├── GradNetTwinRK4.h     # Physics engine (RK4) using manual math
│   │   │   ├── gradnet.h            # Manual Neural Network implementation
│   │   │   └── weights_raw.h        # Hardcoded weights for manual version