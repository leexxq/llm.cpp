#pragma once
namespace gpt2cuda {
    void BatchLayerNormForward(float * outputs , float * means, float * rstds,float const * inputs,float const * gamma,float const * beta,int B,int T,int C);
    void BatchLayerNormBackward(float * d_inputs , float * d_gamma, float * d_beta, float const * d_outputs, float const * inputs, float const * gamma, float const * means, float const * rstds, int B, int T, int C);
}