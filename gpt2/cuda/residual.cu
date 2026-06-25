#include "residual.cuh"
#include "cutlass/util/device_memory.h"
#include "global.cuh"

using DAlloc = cutlass::device_memory::allocation<float>;


__global__ void ResidualForwardKernel(float* outputs , float const * inputs1 , float const * inputs2,int length){
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < length){
        outputs[idx] = inputs1[idx] + inputs2[idx]; 
    }
    
}

__global__ void ResidualBackwardKernel(float* d_inputs1,float* d_inputs2,float const * d_outputs,int length){
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < length){
        d_inputs1[idx] += d_outputs[idx];
        d_inputs2[idx] += d_outputs[idx];
    }
    
}

template<int threads=256>
void ResidualForwardCUDA(float* outputs , float const * inputs1 , float const * inputs2,int length){
    int blocks = (length + threads - 1)/ threads;
    ResidualForwardKernel<<<blocks,threads>>>(outputs, inputs1, inputs2, length);
    CUDA_CHECK_LAST();
}

template<int threads=256>
void ResidualBackwardCUDA(float* d_inputs1,float* d_inputs2,float const * d_outputs,int length){
    int blocks = (length + threads - 1)/ threads;
    ResidualBackwardKernel<<<blocks,threads>>>(d_inputs1,d_inputs2,d_outputs,length);
    CUDA_CHECK_LAST();
}

void gpt2cuda::BatchResidualForward(float* outputs , float const * inputs1 , float const * inputs2,int B,int T,int C){
    //inputs
    DAlloc inputs1_d(B*T*C);
    DAlloc inputs2_d(B*T*C);

    inputs1_d.copy_from_host(inputs1);
    inputs2_d.copy_from_host(inputs2);
    
    //outputs
    DAlloc outputs_d(B*T*C);
    ResidualForwardCUDA<>(outputs_d.get(), inputs1_d.get(), inputs2_d.get(),B*T*C);
    outputs_d.copy_to_host(outputs);
}
void gpt2cuda::BatchResidualBackward(float* d_inputs1 , float * d_inputs2 , float const* d_outputs,int B,int T,int C){
    DAlloc d_inputs1_d(B*T*C);
    DAlloc d_inputs2_d(B*T*C);
    d_inputs1_d.copy_from_host(d_inputs1);
    d_inputs2_d.copy_from_host(d_inputs2);

    DAlloc d_outputs_d(B*T*C);
    ResidualBackwardCUDA(d_inputs1_d.get(), d_inputs2_d.get(), d_outputs_d.get(),B*T*C);

    d_inputs1_d.copy_to_host(d_inputs1);
    d_inputs2_d.copy_to_host(d_inputs2);
}

