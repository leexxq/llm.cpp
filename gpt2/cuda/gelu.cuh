#pragma once
#include "devvector.cuh"
namespace gpt2cuda {
    void BatchGeluForward(float * outputs,float const * inputs, int B,int T, int C);
    void BatchGeluBackward(float * d_inputs,float const * d_outputs,float const* inputs,int B,int T,int C);



    void BatchGeluForward(DevVecf& outputs,const DevVecf& inputs, int B,int T, int C,cudaStream_t stream);
    void BatchGeluBackward(DevVecf& d_inputs,const DevVecf& d_outputs,const DevVecf& inputs,int B,int T,int C,cudaStream_t stream);
}