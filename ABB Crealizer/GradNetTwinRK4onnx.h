#ifndef GRADNET_TWIN_RK4ONNX_H
#define GRADNET_TWIN_RK4ONNX_H

#define _USE_MATH_DEFINES
#include <cmath>

extern "C" {
    void entry(const float flux_input[1][2], float current_output[1][2]);
}

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

class GradNetTwinRK4onnx {
public:
    const float Rs = 0.63f;        
    const int n_p = 2;             
    const float dt = 250e-6f; 

    float psi_d = 0.0f; 
    float psi_q = 0.0f;
    float i_d = 0.0f;   
    float i_q = 0.0f;   

    float get_id() const { return i_d; }
    float get_iq() const { return i_q; }

    // helper function bridges Resuklts logic to the ONNX entry function
    void predict_with_onnx(float pd, float pq, float* out_id, float* out_iq) {
        float input[1][2] = { {pd, pq} };
        float output[1][2] = { {0.0f, 0.0f} };
        
        // Call the generated ONNX
        entry(input, output);
        
        *out_id = output[0][0];
        *out_iq = output[0][1];
    }

    struct Derivatives { float dd; float dq; };

    void update_step(float Vd, float Vq, float rpm) {
        float w_e = (float)n_p * rpm * (2.0f * M_PI / 60.0f);

        // RK4 Slope Calculations calling the ONNX model
        Derivatives k1 = get_derivatives(psi_d, psi_q, Vd, Vq, w_e);
        Derivatives k2 = get_derivatives(psi_d + 0.5f * dt * k1.dd, psi_q + 0.5f * dt * k1.dq, Vd, Vq, w_e);
        Derivatives k3 = get_derivatives(psi_d + 0.5f * dt * k2.dd, psi_q + 0.5f * dt * k2.dq, Vd, Vq, w_e);
        Derivatives k4 = get_derivatives(psi_d + dt * k3.dd, psi_q + dt * k3.dq, Vd, Vq, w_e);

        psi_d += (dt / 6.0f) * (k1.dd + 2.0f * k2.dd + 2.0f * k3.dd + k4.dd);
        psi_q += (dt / 6.0f) * (k1.dq + 2.0f * k2.dq + 2.0f * k3.dq + k4.dq);

        predict_with_onnx(psi_d, psi_q, &i_d, &i_q);
    }

private:
    Derivatives get_derivatives(float pd, float pq, float Vd, float Vq, float we) {
        float cid, ciq;
        predict_with_onnx(pd, pq, &cid, &ciq);
        return { Vd - Rs * cid + we * pq, Vq - Rs * ciq - we * pd };
    }
};

#endif
