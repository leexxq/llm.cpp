#pragma once
namespace gpt2cuda {
    void BatchResidualForward(float* outputs , float const * inputs1 , float const * inputs2, int B, int T, int C);
    void BatchResidualBackward(float* d_inputs1 , float * d_inputs2 , float const* d_outputs,int B,int T,int C);
}