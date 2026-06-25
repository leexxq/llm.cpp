#pragma once
namespace gpt2cuda {
    void BatchSoftmaxForward(float* outputs , float * const  inputs,int B,int T,int V,int Vp);
    void BatchSoftmaxBackward(float* d_inputs, float * const d_outputs,float const * inputs , int B ,int T ,int V,int Vp);
}