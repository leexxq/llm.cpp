#include "cutlass/util/device_memory.h"
#include "gelu.cuh"
#include <cmath>
#include <cstdlib>
#include "error.cuh"


namespace  gpt2cuda{
namespace kernel{
    constexpr  float kconstant1 = 0.7978845608f; //sqrt(2/pi)
    constexpr  float kconstant2 = 0.044715f;

    __global__ void GeluBackwardKernel(float * d_inputs,float const * d_outputs,float const* inputs,int const length){
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
    void GeluBackwardCuda(float * d_inputs,float const * d_outputs,float const* inputs,int length){
        const int blocks = (length + threads - 1)/ threads;
        GeluBackwardKernel<<<blocks,threads>>>(d_inputs, d_outputs, inputs,length);
        CUDA_CHECK_LAST();
    }

    __global__ void  GeluForwardKernel(float * outputs,float const * inputs, int length){
        
        const int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if(idx  < length){
            const float x = inputs[idx];
            const float x_cube = x * x *x;
            const float tanh_arg = kconstant1 * ( x+  kconstant2 * x_cube);
            outputs[idx] = 0.5f * x * (1 + tanh(tanh_arg));
        }

    }

    //params' memory on device 
    template<int threads = 256>
    void GeluForwardCuda(float * outputs,float const * inputs, int length){
        const int blocks = (length + threads - 1)/ threads;
        GeluForwardKernel<<<blocks,threads>>>(outputs, inputs,length);
        CUDA_CHECK_LAST();
    }
} 

    using DAlloc = cutlass::device_memory::allocation<float>;

    void BatchGeluBackward(float * d_inputs,float const * d_outputs,float const* inputs,int B,int T,int C){
        DAlloc d_inputs_d(B*T*C);
        DAlloc d_outputs_d(B*T*C);
        DAlloc inputs_d(B*T*C);
        d_inputs_d.copy_from_host(d_inputs);
        d_outputs_d.copy_from_host(d_outputs);
        inputs_d.copy_from_host(inputs);
        kernel::GeluBackwardCuda<>(d_inputs_d.get(),d_outputs_d.get(),inputs_d.get(),B*T*C);

        d_inputs_d.copy_to_host(d_inputs);
    }


    void BatchGeluForward(float * outputs,float const * inputs, int B,int T, int C){
        DAlloc inputs_d(B*T*C);
        DAlloc outputs_d(B*T*C);

        inputs_d.copy_from_host(inputs);
        
        kernel::GeluForwardCuda(outputs_d.get(), inputs_d.get(), B*T*C);

        outputs_d.copy_to_host(outputs);
    }
}

