#include "cross_entropy.cuh"
#include "cuda/global.cuh"
#include "cutlass/util/device_memory.h"

__global__ void CrossEntropySoftmaxKernel(float * d_logits, float const * probs,int const * targets,int B,int T,int C,float scale){

    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < B*T*C){
        bool pred = targets[idx / C] == (idx % C);
        d_logits[idx] += probs[idx];

        if(pred){
            d_logits[idx] -= 1;
        }
        d_logits[idx] /= scale;
    }

}

template<int threads=256>
void CrossEntropySoftmaxCuda(float * d_logits, float const * probs,int const * targets,int B,int T,int C){

    CrossEntropySoftmaxKernel<<<(B*T*C + threads - 1 )/ threads,threads>>>(d_logits, probs, targets, B, T, C, B*T);
    CUDA_CHECK_LAST();
}

using DAlloc = cutlass::device_memory::allocation<float>;
void CrossEntropySoftmaxBackward(float * d_logits, float const * probs,int const * targets,int B,int T,int C){

    DAlloc d_logits_d;
    DAlloc probs_d;
    cutlass::device_memory::allocation<int> targets_d;

    d_logits_d.copy_from_host(d_logits);
    probs_d.copy_from_host(probs);
    targets_d.copy_from_host(targets);

    CrossEntropySoftmaxCuda(d_logits_d.get(), probs_d.get(), targets_d.get(), B,T,C);

    d_logits_d.copy_to_host(d_logits);
}