#include "cutlass/util/device_memory.h"
#include "gelu.cuh"
#include <cmath>
#include <cstdlib>
#include "global.cuh"



constexpr  float kconstant1 = 0.7978845608f; //sqrt(2/pi)
constexpr  float kconstant2 = 0.044715f;
__global__ void GeluBackwardCuda(float * d_inputs,float const * d_outputs,float const* inputs,int const length){
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx  < length){
        const float x = inputs[idx];
        const float x_square = x * x;
        const float x_cube = x_square * x;
        const float cube = kconstant2 *x_cube;
        const float tanh_arg  = kconstant1 * (x + cube);
        const float tanhf_out = tanh(tanh_arg);
        const float coshf_out = cosh(tanh_arg);
        const float sech_out = 1.0f /(coshf_out * coshf_out);
        const float local_grad = 0.5f * (1.0f + tanhf_out) + x * 0.5f *sech_out * kconstant1 *(1.0f + 3.0f * kconstant2 * x_square);
        d_inputs[idx] += d_outputs[idx] * local_grad;
    }
}


//params' memory on device 
template<int threads = 256>
void GeluBackward(float * d_inputs,float const * d_outputs,float const* inputs,int length){
    const int blocks = (length + threads - 1)/ threads;
    GeluBackwardCuda<<<blocks,threads>>>(d_inputs, d_outputs, inputs,length);
    CUDA_CHECK_LAST();
}



void gpt2cuda::BatchGeluBackward(float * d_inputs,float const * d_outputs,float const* inputs,int B,int T,int C){
    using DAlloc = cutlass::device_memory::allocation<float>;
    DAlloc d_inputs_d(B*T*C);
    DAlloc d_outputs_d(B*T*C);
    DAlloc inputs_d(B*T*C);
    d_inputs_d.copy_from_host(d_inputs);
    d_outputs_d.copy_from_host(d_outputs);
    inputs_d.copy_from_host(inputs);
    GeluBackward<>(d_inputs_d.get(),d_outputs_d.get(),inputs_d.get(),B*T*C);

    d_inputs_d.copy_to_host(d_inputs);
}