#pragma once
namespace gpt2cuda {
    void BatchEncoderForward(float * outputs,int const* inputs ,float  const * wte,float const*  wpe,int B,int T,int C,int V ,int MaxT);
    void BatchEncoderBackward(float* d_wte,float * d_wpe, float const*  d_outputs,int const * inputs,int B,int T,int C,int Vp ,int MaxT);
}