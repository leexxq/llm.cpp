#pragma once
#include "devvector.cuh"
namespace gpt2cuda {
    void BatchEncoderForward(float * outputs,int const* inputs ,float  const * wte,float const*  wpe,int B,int T,int C,int V ,int MaxT);
    void BatchEncoderBackward(float* d_wte,float * d_wpe, float const*  d_outputs,int const * inputs,int B,int T,int C,int Vp ,int MaxT);

    void BatchEncoderForward(DevVecf& outputs,const DevVeci& inputs ,const DevVecf& wte,const DevVecf& wpe,int B,int T,int C,int V ,int MaxT,cudaStream_t stream);
    void BatchEncoderBackward(DevVecf& d_wte,DevVecf& d_wpe, const DevVecf& d_outputs,const DevVeci& inputs,int B,int T,int C,int Vp ,int MaxT,cudaStream_t stream);
}