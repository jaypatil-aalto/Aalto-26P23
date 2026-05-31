#ifndef GRADNET_TWIN_RK4_H
#define GRADNET_TWIN_RK4_H

#define _USE_MATH_DEFINES
#include "gradnet.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

class GradNetTwinRK4 {
public:
    const float Rs = 0.63f;        // Stator Resistance
    const int n_p = 2;             // Pole Pairs
    const float dt = 1.0f / 4000.0f; // 250us cycle

    float psi_d = 0.0f, psi_q = 0.0f;
    float i_d = 0.0f, i_q = 0.0f;

    struct Derivatives { float dd; float dq; };

    float get_id() const { return i_d; }
    float get_iq() const { return i_q; }

    void update_step(float Vd, float Vq, float rpm) {
        float w_e = n_p * rpm * (2.0f * (float)M_PI / 60.0f);

        // Standard Runge-Kutta 4th Order Implementation
        // RK4 Step 1
        Derivatives k1 = get_derivatives(psi_d, psi_q, Vd, Vq, w_e);
        
        // RK4 Step 2
        Derivatives k2 = get_derivatives(psi_d + 0.5f*dt*k1.dd, 
            psi_q + 0.5f*dt*k1.dq, Vd, Vq, w_e);
        
        // RK4 Step 3
        Derivatives k3 = get_derivatives(psi_d + 0.5f*dt*k2.dd, 
            psi_q + 0.5f*dt*k2.dq, Vd, Vq, w_e);
        
        // RK4 Step 4
        Derivatives k4 = get_derivatives(psi_d + dt*k3.dd, 
            psi_q + dt*k3.dq, Vd, Vq, w_e);

        // Final weighted update
        psi_d += (dt / 6.0f) * (k1.dd + 2.0f * k2.dd + 2.0f * k3.dd+ k4.dd);
        psi_q += (dt / 6.0f) * (k1.dq + 2.0f * k2.dq + 2.0f * k3.dq + k4.dq);

        // Update public-facing currents
        Result current = GradNet::predict(psi_d,psi_q);
    
        i_d = current.d;
        i_q = current.q;
}
private:
    Derivatives get_derivatives(float pd, float pq, float Vd, float Vq, float we){
        Result curr = GradNet::predict(pd,pq);
        // dPsi/dt = V - Rs*i + w*Psi_orthogonal
        float dpsi_d = Vd - Rs * curr.d + we * pq;
        float dpsi_q = Vq - Rs * curr.q - we * pd;
        return { dpsi_d, dpsi_q };
    }
};
#endif