#include "cross_entropy.cuh"
#include "error.cuh"
#include "cutlass/util/device_memory.h"

namespace gpt2cuda {
namespace kernel {
    __global__ void CrossEntropySoftmaxBackwardKernel(float * d_logits, float const * probs,int const * targets,int ld ,int length,int stride ,float scale){

        const int idx = threadIdx.x + blockDim.x * blockIdx.x;

        if(idx < length){
            bool pred = targets[idx/stride] == (idx % stride);
            const int offest = (idx/stride)*ld + idx%stride;
            float val = d_logits[offest];

            val += probs[offest];

            if(pred){
               val -= 1;
            }

            d_logits[offest] =(d_logits[offest] + val) * scale;
        }

    }
    __global__ void CrossEntropyForwardKernel(float* losses, float const * inputs, int const * targets,int length,int stride){

        const int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if(idx < length){
            losses[idx] = -log(inputs[idx * stride + targets[idx]]);
        }
    }

    template<int threads=256>
    void CrossEntropySoftmaxBackwardCUDA(float * d_logits, float const * probs,int const * targets,int ld , int length,int stride ,float scale){
        CrossEntropySoftmaxBackwardKernel<<<(length + threads - 1 )/ threads,threads>>>(d_logits, probs, targets, ld,length,stride , scale);
        CUDA_CHECK_LAST();
    }

    template<int threads=256>
    void CrossEntropyForwardCUDA(float* losses ,  float const * inputs, int const * targets,int length,int stride){
        CrossEntropyForwardKernel<<<(length + threads - 1 )/ threads,threads>>>(losses,inputs ,targets, length,stride);
        CUDA_CHECK_LAST();
    }
}

    using DAllocf = cutlass::device_memory::allocation<float>;
    using DAlloci = cutlass::device_memory::allocation<int>;
    void BatchCrossEntropySoftmaxBackward(float * d_logits, float const * probs,int const * targets,int B,int T,int V,int Vp,float scale){

        DAllocf d_logits_d(B*T*Vp);
        DAllocf probs_d(B*T*Vp);
        DAlloci targets_d(B*T);

        d_logits_d.copy_from_host(d_logits);
        probs_d.copy_from_host(probs);
        targets_d.copy_from_host(targets);

        kernel::CrossEntropySoftmaxBackwardCUDA(d_logits_d.get(), probs_d.get(), targets_d.get(), Vp,B*T*V,V,scale);

        d_logits_d.copy_to_host(d_logits);
    }

    void BatchCrossEntropyForward(float* losses, float const * inputs, int const * targets,int B,int T,int V){
        DAllocf inputs_d(B*T*V);
        DAllocf losses_d(B*T);
        DAlloci targets_d(B*T);

        inputs_d.copy_from_host(inputs);
        targets_d.copy_from_host(targets);

        kernel::CrossEntropyForwardCUDA(losses_d.get(), inputs_d.get(), targets_d.get(), B*T, V);

        losses_d.copy_to_host(losses);

    }


}
