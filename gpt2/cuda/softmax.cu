#include "softmax.cuh"
#include "global.cuh"
#include <cassert>
#include "cutlass/util/device_memory.h"
#include <cmath>
#include <limits>

namespace gpt2cuda {
namespace kernel{

    __global__ void SoftmaxforwardKernel(float * outputs, float * const inputs , int ld , int length,int stride){
        const int bidx = blockIdx.x;
        const int warpidx = threadIdx.x / warpSize;
        assert(blockDim.x % warpSize==0);
        const int warps = blockDim.x / warpSize;
        const int sum_idx = (bidx * warps + warpidx);
        const int sum_offest = sum_idx *ld ;

        bool out_length_pred = (sum_idx >= (length / stride));
        if(out_length_pred) {return;}

        const int lane = threadIdx.x%warpSize;
        float sum = 0;
        float maxval = std::numeric_limits<float>::min();


        //load stride sum each thread;
        for(int i = 0 ; i < stride / warpSize ; ++ i){
            const float val = inputs[sum_offest + lane + i * warpSize];
            maxval = max(maxval,val);
            sum += exp(val);
        }

        // residual
        bool residual_pred = (lane < stride % warpSize);
        if(residual_pred){
            const float val = inputs[sum_offest + stride - 1 - lane];
            maxval = max(maxval,val);
            sum += exp(val);
        }

        // printf("lane : %d, warpidx : %d , %f\n",lane,warpidx,mean);
        
        //warp level reduction
        uint32_t mask = 0xFFFFFFFFU;
        for(int offest = warpSize/2 ; offest > 0 ; offest/= 2){
            sum += __shfl_down_sync(mask,sum,offest);
            maxval = max(__shfl_down_sync(mask,sum,offest),maxval);
        }
        bool first_pred = (lane == 0);
        if(first_pred){
            sum *= exp(-maxval);
        }
        //brodcast sum and maxval to warp other threads
        sum = __shfl_sync(mask,sum,0);
        maxval = __shfl_sync(mask,maxval,0);

        // printf("lane : %d, warpidx : %d , %f\n",lane,warpidx,mean);
        if(residual_pred){
            const int local_offest = stride - 1 - lane;
            const float val = inputs[sum_offest + local_offest];
            outputs[sum_offest + local_offest] = exp(val - maxval)/sum;
        }

        //load stride sum each thread;
        for(int i = 0 ; i < stride / warpSize ; ++ i){
            const int local_offest = lane + i * warpSize;
            const float val = inputs[sum_offest + local_offest];
            outputs[sum_offest + local_offest] = exp(val - maxval)/sum;
        }
        
    }


    template <int threads= 256>
    void SoftmaxForwardCUDA(float * outputs, float * const inputs ,int ld , int length,int stride){

        //compute one row by warps
        assert(length % stride == 0);
        constexpr int warp_size  = 32;
        constexpr int warps = threads / warp_size;
        const int blocks  = ((length/stride) + warps -1 )/ (warps);

        // std::cout <<"blocks:" << blocks << std::endl;
        SoftmaxforwardKernel<<<blocks,threads>>>(outputs,inputs,ld,length,stride); 
        CUDA_CHECK_LAST();
    }


}

    using DAlloc = cutlass::device_memory::allocation<float>;
    void BatchSoftmaxForward(float* outputs , float * const  inputs,int B,int T,int V,int Vp){
        DAlloc outputs_d(B*T*Vp);
        DAlloc inputs_d(B*T*Vp);

        inputs_d.copy_from_host(inputs);

        kernel::SoftmaxForwardCUDA(outputs_d.get(), inputs_d.get(),Vp,B*T*V ,V);

        outputs_d.copy_to_host(outputs);

    }

    void BatchSoftmaxBackward(float* d_inputs, float * const d_outputs,float const * inputs , int B ,int T ,int V ,int Vp){

    }
}
