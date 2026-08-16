#pragma once
#include "devvector.cuh"

namespace gpt2cuda {
    void BatchCrossEntropySoftmaxBackward(float * d_logits, float const * probs,int const * targets,int B,int T,int V,int Vp,float scale);
    void BatchCrossEntropyForward(float* losses, float const * inputs, int const * targets,int B,int T,int Vp);

    void BatchCrossEntropySoftmaxBackward(DevVecf& d_logits, const DevVecf& probs,const DevVeci& targets,int B,int T,int V,int Vp,float scale,cudaStream_t stream);
    void BatchCrossEntropyForward(DevVecf& losses, const DevVecf& inputs, const DevVeci& targets,int B,int T,int Vp,cudaStream_t stream);


// void CrossEntropyForward( float* losses , float const * probs,int const * targets);
}