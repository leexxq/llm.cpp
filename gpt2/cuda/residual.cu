#include "residual.cuh"
#include "cutlass/util/device_memory.h"
// #include <__clang_cuda_builtin_vars.h>


using DAlloc = cutlass::device_memory::allocation<float>;


__global__ void ResidualKernel(float* outputs , float const * inputs1 , float const * inputs2,int length){
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < length){
        outputs[idx] = inputs1[idx] + inputs2[idx]; 
    }
    
}

template<int threads=256>
void Residual1D(float* outputs , float const * inputs1 , float const * inputs2,int length){
    int blocks = (length + threads - 1)/ threads;
    ResidualKernel<<<blocks,threads>>>(outputs, inputs1, inputs2, length);
}

void gpt2cuda::BatchResidualForward(float* outputs , float const * inputs1 , float const * inputs2,int B,int T,int C){
    //inputs
    DAlloc inputs1_d(B*T*C);
    DAlloc inputs2_d(B*T*C);

    inputs1_d.copy_from_host(inputs1);
    inputs2_d.copy_from_host(inputs2);
    
    //outputs
    DAlloc outputs_d(B*T*C);
    Residual1D<>(outputs_d.get(), inputs1_d.get(), inputs2_d.get(),B*T*C);
    outputs_d.copy_to_host(outputs);
}
void gpt2cuda::BatchResidualBackward(float* d_inputs , float const* d_outputs,int B,int T,int C){
    
}

