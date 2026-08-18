#include "softmax.cuh"
#include "error.cuh"
#include <cassert>
#include "cutlass/util/device_memory.h"
#include <cmath>
#include <limits>
#include <math_constants.h>

namespace gpt2cuda {
namespace kernel{

    template<int threads=256>
    __global__ void SoftmaxforwardKernel(float * outputs, float const * inputs , int ld , int length,int stride){
        const int bidx = blockIdx.x;
        const int warpidx = threadIdx.x / warpSize;

        assert(blockDim.x % warpSize==0);

        const int  warps = threads / warpSize;
        const int row_idx = (bidx * warps + warpidx);
        const int row_offest = row_idx *ld ; //current warp compute row



        bool out_length_pred = (row_idx >= (length / stride));
        if(out_length_pred) {return;}

        const int lane = threadIdx.x%warpSize;
        float sum = 0;
        float maxval = -CUDART_INF_F;


        //compute stride maxval each thread;
        for(int i = 0 ; i < stride / warpSize ; ++ i){
            const float val = inputs[row_offest + lane + i * warpSize];
            maxval = max(maxval,val);
        }

        // residual
        bool residual_pred = (lane < stride % warpSize);
        if(residual_pred){
            const float val = inputs[row_offest + stride - 1 - lane];
            maxval = max(maxval,val);
        }


        constexpr uint32_t full_mask = 0xFFFFFFFFU;
        for(int offest = warpSize/2 ; offest > 0 ; offest>>= 1){
            maxval = max(__shfl_xor_sync(full_mask,maxval,offest),maxval);
        }

        //compute stride sum each thread;
        for(int i = 0 ; i < stride / warpSize ; ++i){
            const float val = inputs[row_offest + lane + i * warpSize];
            sum += expf(val - maxval);
        }

        // residual
        if(residual_pred){
            const float val = inputs[row_offest + stride - 1 - lane];
            sum += expf(val - maxval);
        }
        
        
        for(int offest = warpSize/2 ; offest > 0 ; offest >>= 1){
            sum += __shfl_xor_sync(full_mask,sum,offest);
        }

        // if(lane ==0){
        //     printf("lane : %d, warpidx : %d , {%f,%f}\n",lane,warpidx,maxval,sum);
        // }


        if(residual_pred){
            const int local_offest = stride - 1 - lane;
            const float val = inputs[row_offest + local_offest];
            outputs[row_offest + local_offest] = expf(val - maxval)/sum;
        }

        //load stride sum each thread;
        for(int i = 0 ; i < stride / warpSize ; ++i){
            const int local_offest = lane + i * warpSize;
            const float val = inputs[row_offest + local_offest];
            outputs[row_offest + local_offest] = expf(val - maxval)/sum;
        }
        // if(threadIdx.x ==0){
        //     printf("bidx: %d , lane : %d, warpidx : %d , row : %d \n",bidx,lane,warpidx,row_idx);
        // }
        
    }


    template <int threads= 256>
    void SoftmaxForwardCUDA(float * outputs, float const *  inputs ,int ld , int length,int stride, cudaStream_t stream = 0){

        //compute one row by warps
        assert(length % stride == 0);
        constexpr int warp_size  = 32;
        constexpr int warps = threads / warp_size;
        const int blocks  = ((length/stride) + warps -1 )/ (warps);

        // std::cout <<"blocks:" << blocks << std::endl;
        
        SoftmaxforwardKernel<threads><<<blocks,threads,0,stream>>>(outputs,inputs,ld,length,stride); 
        CUDA_CHECK_LAST();
    }


}

    using DAlloc = cutlass::device_memory::allocation<float>;
    void BatchSoftmaxForward(float* outputs , float * const  inputs,size_t B,size_t T,size_t V,size_t Vp){
        DAlloc outputs_d(B*T*Vp);
        DAlloc inputs_d(B*T*Vp);

        inputs_d.copy_from_host(inputs);

        kernel::SoftmaxForwardCUDA(outputs_d.get(), inputs_d.get(),Vp,B*T*V ,V);

        outputs_d.copy_to_host(outputs);

    }

    void BatchSoftmaxBackward(float* d_inputs, float * const d_outputs,float const * inputs , size_t B ,size_t T ,size_t V ,size_t Vp){

    }




    void BatchSoftmaxForward(DevVecf& outputs, const DevVecf& inputs, size_t B, size_t T, size_t V, size_t Vp, cudaStream_t stream){
        kernel::SoftmaxForwardCUDA(outputs.data(), inputs.data(),Vp,B*T*V ,V,stream);
    }
    void BatchSoftmaxBackward(DevVecf& d_inputs, const DevVecf& d_outputs, const DevVecf& inputs, size_t B, size_t T, size_t V, size_t Vp, cudaStream_t stream){


    }
}
