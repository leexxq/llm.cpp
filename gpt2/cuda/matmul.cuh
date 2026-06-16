#pragma once

namespace gpt2cuda {
    void BatchMatmulForward(float * outputs,float const *  inputs , float const * weight, float const * bias , int B, int T,int  C,int Oc);
    void BatchMatmulBackward(float * d_inputs, float* d_weight, float* d_bias,float const * d_outputs, float const *  inputs , float const * weight,int B, int T,int  C,int Oc);

    void BatchMatmulGeluForward(float * outputs,float const *  inputs , float const * weight, float const * bias , int B, int T,int  C,int Oc);
    void BatchMatmulGeluBackward(float * d_inputs, float* d_weight, float* d_bias,float const * d_outputs, float const *  inputs , float const * weight,int B, int T,int  C,int Oc);
}


// VecBTC CudaMatmul(const VecBTC& inputs,const Matf& weight,const Vecf& bias);