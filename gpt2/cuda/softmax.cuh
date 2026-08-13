#pragma once
#include <cstddef>
#include "devvector.cuh"
namespace gpt2cuda {
    void BatchSoftmaxForward(float* outputs , float * const  inputs,size_t B,size_t T,size_t V,size_t Vp);
    void BatchSoftmaxBackward(float* d_inputs, float * const d_outputs,float const * inputs , size_t B ,size_t T ,size_t V,size_t Vp);

    void BatchSoftmaxForward(DevVecf& outputs, const DevVecf& inputs, size_t B, size_t T, size_t V, size_t Vp, cudaStream_t stream);
    void BatchSoftmaxBackward(DevVecf& d_inputs, const DevVecf& d_outputs, const DevVecf& inputs, size_t B, size_t T, size_t V, size_t Vp, cudaStream_t stream);


}