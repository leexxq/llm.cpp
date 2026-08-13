#pragma once
#include "devvector.cuh"
namespace gpt2cuda {
    void BatchLayerNormForward(float * outputs , float * means, float * rstds,float const * inputs,float const * gamma,float const * beta,int B,int T,int C);
    void BatchLayerNormBackward(float * d_inputs , float * d_gamma, float * d_beta, float const * d_outputs, float const * inputs, float const * gamma, float const * means, float const * rstds, int B, int T, int C);

    void BatchLayerNormForward(DevVecf& outputs, DevVecf& means, DevVecf& rstds, const DevVecf& inputs, const DevVecf& gamma, const DevVecf& beta, int B, int T, int C, cudaStream_t stream);
    void BatchLayerNormBackward(DevVecf& d_inputs, DevVecf& d_gamma, DevVecf& d_beta, const DevVecf& d_outputs, const DevVecf& inputs, const DevVecf& gamma, const DevVecf& means, const DevVecf& rstds, int B, int T, int C, cudaStream_t stream );
}