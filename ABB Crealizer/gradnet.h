#ifndef GRADNET_H
#define GRADNET_H

#include <cmath>
#include "weights_raw.h"

//torch.Tensor([d,q])
struct Result { float d, q; };

class GradNet {
public:
    /**
     * Python: forward(self,i_s)
     * flux -> current
     * symmetry logic: 0.5 * (g(x) + g(x @ C) @ C)
     */
    static Result predict(float psi_d, float psi_q) {
        //normalize inputs
        Result x = {psi_d / RAW_PSI_BASE, psi_q/RAW_PSI_BASE };

        //1st pass of g(x)
        Result g1= function_g(x);

        //2nd pass, Python: i_s @ self.C
        Result x_sym = {x.d,-x.q};
        Result g2 = function_g(x_sym);
        g2.q = -g2.q;
        
        //average and de-normalize
        return {
            0.5f * (g1.d + g2.d) * RAW_I_BASE,
            0.5f * (g1.q + g2.q) * RAW_I_BASE
        };
    }

private:
    /**
     * python _g(self,x)
     */
    static Result function_g(Result x) {
        //RAW_MU - pre-calculated
        Result out;
        //no matrix math, simple calucaltions
        out.d = (x.d * RAW_MU[0]) + RAW_B0[0];
        out.q = (x.q * RAW_MU[1]) + RAW_B0[1];

        //all 12 neurons at once
        for (int i = 0; i < RAW_N; ++i) {
            //dot product of xd and xq and A[i]
            float z = (x.d * RAW_A[i][0] + x.q * RAW_A[i][1]) + RAW_B[i];
            //sigmoid function
            float act = z / std::sqrt(z * z + RAW_BETA);
            
            out.d += act * RAW_A[i][0];
            //summaraizing the 12 neurons
            out.q += act * RAW_A[i][1];
        }
        return out;
    }
};
#endif