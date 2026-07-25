#pragma once
#include <cstddef>
namespace gpt2cuda {
    void BatchSoftmaxForward(float* outputs , float * const  inputs,size_t B,size_t T,size_t V,size_t Vp);
    void BatchSoftmaxBackward(float* d_inputs, float * const d_outputs,float const * inputs , size_t B ,size_t T ,size_t V,size_t Vp);
}