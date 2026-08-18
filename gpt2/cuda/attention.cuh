#pragma once
#include "devvector.cuh"
namespace gpt2cuda {
    void BatchAttentionForward(float * outputs ,float * logsumexp ,  float const * inputs,int B,int T ,int C3,int NH);
    void BatchAttentionBackward(float *d_inputs,float const *d_outputs,float const* outputs, float const *inputs,float const* logsumexp ,int B, int T, int C3, int NH);
    void BatchCausalAttentionForward(float * outputs,float * logsumexp  , float const * inputs,int B,int T ,int C3,int NH);
    void BatchCausalAttentionBackward(float *d_inputs,float const *d_outputs,float const* outputs, float const *inputs,float const* logsumexp ,int B, int T, int C3, int NH);


    void BatchCausalAttentionForward(DevVecf& outputs,DevVecf& logsumexp  , const DevVecf& inputs,int B,int T ,int C3,int NH,cudaStream_t stream);
    void BatchCausalAttentionBackward(DevVecf& d_inputs,DevVecf& att_D , const DevVecf& d_outputs, const DevVecf& outputs, const DevVecf& inputs,const DevVecf& logsumexp ,int B, int T, int C3, int NH,cudaStream_t stream);
}