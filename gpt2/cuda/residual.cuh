#pragma once
#include "devvector.cuh"
namespace gpt2cuda {
    void BatchResidualForward(float* outputs , float const * inputs1 , float const * inputs2, int B, int T, int C);
    void BatchResidualBackward(float* d_inputs1 , float * d_inputs2 , float const* d_outputs,int B,int T,int C);

    void BatchResidualForward(DevVecf& outputs, const DevVecf& inputs1, const DevVecf& inputs2, size_t B, size_t T, size_t C,cudaStream_t stream);
    void BatchResidualBackward(DevVecf& d_inputs1, const DevVecf& d_inputs2, const DevVecf& d_outputs, size_t B, size_t T, size_t C,cudaStream_t stream);
}