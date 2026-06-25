#include "cute/util/debug.hpp"
#include "cutlass/util/device_memory.h"
#include "layernorm.cuh"
#include "global.cuh"
#include <cub/cub.cuh>
#include <cassert>
#include <cstdint>
namespace gpt2cuda {
namespace kernel {
    constexpr float eplison = 1e-5;
    __global__ void LayerNormForwardKernel(float * outputs_d,float * means , float* rstds , float const * inputs_d,float const * gamma , float const * beta, int const length,int const stride){
        const int bidx = blockIdx.x;
        const int warpidx = threadIdx.x / warpSize;
        assert(blockDim.x % warpSize==0);
        const int warps = blockDim.x / warpSize;
        const int means_idx = (bidx * warps + warpidx);
        const int means_offest = means_idx * stride;

        bool out_length_pred = (means_idx >= (length / stride));
        if(out_length_pred) {return;}

        const int lane = threadIdx.x%warpSize;
        float mean = 0;
        float rstd = 0;
        // residual
        bool residual_pred = (lane < stride % warpSize);
        if(residual_pred){
            const float val = inputs_d[means_offest + stride - 1 - lane];
            mean += val;
            rstd += val * val;
        }

        //load stride sum each thread;
        for(int i = 0 ; i < stride / warpSize ; ++ i){
            const float val = inputs_d[means_offest + lane + i * warpSize];
            mean += val;
            rstd += val * val;
        }

        // printf("lane : %d, warpidx : %d , %f\n",lane,warpidx,mean);
        
        //warp level reduction
        constexpr uint32_t full_mask = 0xFFFFFFFFU;
        // for(int offest = warpSize/2 ; offest > 0 ; offest/= 2){
        //     mean += __shfl_down_sync(full_mask,mean,offest);
        //     rstd +=__shfl_down_sync(full_mask,rstd,offest);
        // }
        // bool first_pred = (lane == 0);
        // if(first_pred){
        //     mean /= stride;
        //     // printf("lane : %d, warpidx : %d , %f\n",lane,warpidx,mean);
        //     rstd = rstd / stride - mean * mean;
        //     rstd = 1.0f / sqrtf(rstd + eplison);
        //     means[means_idx] = mean;
        //     rstds[means_idx] = rstd;
        // }
        // //brodcast mean and rstd
        // mean = __shfl_sync(full_mask,mean,0);
        // rstd = __shfl_sync(full_mask,rstd,0);


        for(int offest = warpSize/2 ; offest > 0 ; offest/= 2){
            mean += __shfl_xor_sync(full_mask,mean,offest);
            rstd +=__shfl_xor_sync(full_mask,rstd,offest);
        }


        mean /= stride;
        // printf("lane : %d, warpidx : %d , %f\n",lane,warpidx,mean);
        rstd = rstd / stride - mean * mean;
        rstd = 1.0f / sqrtf(rstd + eplison);

        bool first_pred = (lane == 0);
        if(first_pred){
            // printf("lane : %d, warpidx : %d , %f\n",lane,warpidx,mean);
            means[means_idx] = mean;
            rstds[means_idx] = rstd;
        }


        //calculate layernorm

        // printf("lane : %d, warpidx : %d , %f\n",lane,warpidx,mean);
        if(residual_pred){
            const int local_offest = stride - 1 - lane;
            const float val = inputs_d[means_offest + local_offest];
            outputs_d[means_offest + local_offest] = (val - mean) * rstd * gamma[local_offest]  + beta[local_offest];
        }

        //load stride sum each thread;
        for(int i = 0 ; i < stride / warpSize ; ++ i){
            const int local_offest = lane + i * warpSize;
            const float val = inputs_d[means_offest + local_offest];
            outputs_d[means_offest + local_offest] = (val - mean) * rstd * gamma[local_offest]  + beta[local_offest];
        }
        
    };

    template<int threads=256>
    void LayerNormForwardCUDA(float * outputs_d,float * means , float* rstds , float const * inputs_d,float const * gamma,float const * beta,int const length,int const stride){

        //compute one row by warps
        assert(length % stride == 0);
        assert(stride%32 == 0);
        constexpr int warp_size  = 32;
        constexpr int warps = threads / warp_size;
        const int blocks  = ((length/stride) + warps -1 )/ (warps);

        // std::cout <<"blocks:" << blocks << std::endl;
        LayerNormForwardKernel<<<blocks,threads>>>(outputs_d,means,rstds,inputs_d,gamma,beta,length,stride); 
        CUDA_CHECK_LAST();
    }


    //used for compute d_gamma , d_beta and \hat{x}_{ij}
    __global__ void GammaAndBetaDerivateKernel(float * d_gamma, float * d_beta, float const * d_outputs, float  * hat_inputs,int B, int T, int C){
        const int length = B*T*C;
        const int stride = C;

        const int bidx = blockIdx.x;
        const int tidx = threadIdx.x; 
        // const int idx = tidx + bidx * blockDim.x;

        const int warpidx = tidx / warpSize;
        const int lane = tidx % warpSize;
        assert(blockDim.x%warpSize == 0);
        const int warp_nums = blockDim.x / warpSize;
        const int wave_nums = (length + warpSize * stride - 1)/(warpSize * stride);
        const u_int32_t mask = 0xFFFFFFFFU;


        for(int wave = 0 ; wave < wave_nums ; ++wave){
            const int warp_offest = bidx * warp_nums + warpidx ;
            const int wave_offest = warp_offest+ wave * warpSize * stride;
            const int thread_offest = wave_offest +  lane * stride;
            float d_beta_i = 0,d_gamma_i=0;
            if(thread_offest < length){
                d_beta_i = d_outputs[thread_offest];
                d_gamma_i = d_outputs[thread_offest] * hat_inputs[thread_offest];
            }
            for(int offest = warpSize / 2 ; offest > 0 ; offest/=2){
                d_beta_i += __shfl_down_sync(mask,d_beta_i,offest);
                d_gamma_i += __shfl_down_sync(mask,d_gamma_i,offest);
            }
            if(lane == 0){
                d_beta[warp_offest] += d_beta_i;
                d_gamma[warp_offest] += d_gamma_i;
            }
        }
    }


    // __global__ void DerivativeLayerNorm1DKernel(float * d_inputs , float * d_gamma, float * d_beta, float const * d_outputs, float  * inputs, float const * gamma, float const * beta, float const * means, float const * rstds, int B, int T, int C){


    // }

    __global__ void ComputeHatInputsKernel(float*inputs,float const*means,float const* rstds,int B,int T,int C){

        const int bidx = blockIdx.x;
        const int tidx = threadIdx.x; 
        const int idx = tidx + bidx * blockDim.x;

        if(idx < B*T*C){
            inputs[idx] = (inputs[idx] - means[idx / C]) *rstds[idx / C];
        }

    }




    template<int K>
    __launch_bounds__(256) __global__ void ComputeDerivateInputsKernel(float* d_inputs,float const * d_outputs, float const* inputs,float const* gamma,float const * rstds,int B,int T,int C){

        //used stored d_outputs
        extern __shared__ float s_data[];
        const int bidx = blockIdx.x ;
        const int threads = blockDim.x;
        const int tidx = threadIdx.x;
        assert(threads/warpSize);
        const int warps = threads / warpSize;
        const int lane = tidx % warpSize; 
        const int wid = tidx / warpSize; 
        const int wave = C / threads;
        const u_int32_t mask = 0xFFFFFFFFU;

        // used for reduction 
        static __shared__ float shared[64];
        #pragma unroll 1
        for(int row_off =  0; row_off < K ; ++row_off){
            const int row = bidx * K +row_off;
            if( row >= B*T){
                return;
            }
            float term2 = 0;
            float term3 = 0;
            for (int i =0 ; i< wave ; ++i){
                const int tid_off = tidx + i * threads;
                float val = gamma[tid_off]  * d_outputs[row*C + tid_off];
                term2 += val;
                term3 += inputs[row*C + tid_off] * val;
                s_data[tid_off] = val;
            }
            bool residual_pred = (tidx + wave * threads) < C; 
            if(residual_pred){
                const int tid_off = tidx + wave * threads;
                float val = gamma[tid_off]  * d_outputs[row*C + tid_off];
                term2 += val;
                term3 += inputs[row*C + tid_off] * val;
                s_data[tid_off] = val;
            }
            
            // __syncthreads();
            //block reduction gamma_l * d_outputs_il and gamma_l * d_outputs_il * x_hat_il


            for(int offest = warpSize / 2 ; offest > 0 ; offest/=2){
                term2 += __shfl_down_sync(mask,term2,offest);
                term3 += __shfl_down_sync(mask,term3,offest);
            }

            if(lane == 0){
                shared[wid] = term2;
                shared[wid + warpSize] = term3;
            }

            __syncthreads();

            term2 = (tidx < warps) ? shared[lane] : 0.0f;
            term3 = (tidx < warps) ? shared[lane + warpSize] : 0.0f;

            for(int offest = warpSize / 2 ; offest > 0 ; offest/=2){
                term2 += __shfl_down_sync(mask,term2,offest);
                term3 += __shfl_down_sync(mask,term3,offest);
            }

            if(tidx == 0){
                term2 /= C;
                term3 /= C;
                for(int i =0 ; i < warps ; ++i){
                    shared[i] = term2;
                    shared[i + warps] = term3;
                }
            }

            __syncthreads();

            if(lane ==0){
                term2 = shared[wid];
                term3 = shared[wid + warps];
            }

            term2 = __shfl_sync(mask,term2,0);
            term3 = __shfl_sync(mask,term3,0);


            for (int i =0 ; i< wave ; ++i){
                const int tid_off = tidx + i * threads;
                d_inputs[row * C + tid_off] += rstds[row] * (s_data[tid_off] - term2 - term3 * inputs[row * C + tid_off]);
            }

            residual_pred = (tidx + wave * threads) < C; 

            if(residual_pred){
                const int tid_off = tidx + wave * threads;
                d_inputs[row * C + tid_off] += rstds[row] * (s_data[tid_off] - term2 - term3 * inputs[row * C + tid_off]);
            }

        }

    }

    //require device memory 
    template <int threads=256,int K =1> 
    void LayerNormBackwardCUDA(float * d_inputs , float * d_gamma, float * d_beta, float const * d_outputs, float * inputs, float const * gamma,float const * means, float const * rstds, int B, int T, int C){

        assert(C!=0);
        //convert inputs to  \hat{x}_{ij}
        ComputeHatInputsKernel<<<(B*T*C + threads - 1)/ threads,threads>>>(inputs,means,rstds, B,T,C);
        CUDA_CHECK_LAST();

        CUDA_CHECK(cudaDeviceSynchronize());

        //compute d_gamma and d_beta
        constexpr int warp_threads = 32;
        static_assert(threads % warp_threads == 0);
        constexpr int warp_count = threads / warp_threads;
        // constexpr int threads = warp_threads * warp_count;
        int blocks = (C+ warp_count - 1)/ warp_count;

        GammaAndBetaDerivateKernel<<<blocks,threads>>>(d_gamma, d_beta, d_outputs, inputs,B,T, C);
        CUDA_CHECK_LAST();

        assert((B*T)% K == 0);

        //compute d_inputs
        //compute one row by blocks
        ComputeDerivateInputsKernel<K><<<(B*T + K - 1) / K,threads,sizeof(float) * C>>>(d_inputs, d_outputs , inputs, gamma,rstds, B,T,C);
        CUDA_CHECK_LAST();

    }

}

    using DAlloc = cutlass::device_memory::allocation<float>;

    void BatchLayerNormForward(float * outputs , float * means, float * rstds,float const * inputs,float const * gamma,float const * beta,int B,int T,int C){
        // calculate E[x_i] and rstd_i;

        DAlloc outputs_d(B*T*C);
        DAlloc inputs_d(B*T*C);
        DAlloc means_d(B*T);
        DAlloc rstds_d(B*T);
        DAlloc gamma_d(C);
        DAlloc beta_d(C);
        inputs_d.copy_from_host(inputs);
        gamma_d.copy_from_host(gamma);
        beta_d.copy_from_host(beta);

        kernel::LayerNormForwardCUDA<>(outputs_d.get(),means_d.get(),rstds_d.get(),inputs_d.get(),gamma_d.get(),beta_d.get(),B*T*C,C);

        outputs_d.copy_to_host(outputs);
        means_d.copy_to_host(means);
        rstds_d.copy_to_host(rstds);

    }




    void BatchLayerNormBackward(float * d_inputs , float * d_gamma, float * d_beta, float const * d_outputs, float const * inputs, float const * gamma, float const * means, float const * rstds, int B, int T, int C){
        //input
        DAlloc d_outputs_d(B*T*C);
        DAlloc inputs_d(B*T*C);
        DAlloc gamma_d (C);
        DAlloc beta_d  (C);
        DAlloc means_d (B*T);
        DAlloc rstds_d (B*T);


        d_outputs_d .copy_from_host(d_outputs);
        inputs_d    .copy_from_host(inputs);
        gamma_d     .copy_from_host(gamma);
        means_d     .copy_from_host(means);
        rstds_d     .copy_from_host(rstds);

        
        //output
        DAlloc d_inputs_d(B*T*C);
        DAlloc d_gamma_d(C);
        DAlloc d_beta_d(C);
        


        // inputs modified
        kernel::LayerNormBackwardCUDA<>(d_inputs_d.get(), d_gamma_d.get(), d_beta_d.get(),  d_outputs_d.get(),  inputs_d.get(),  gamma_d.get(),   means_d.get(),  rstds_d.get(),  B,  T,  C);

        d_inputs_d.copy_to_host(d_inputs);
        d_gamma_d.copy_to_host(d_gamma);
        d_beta_d.copy_to_host(d_beta);

    }

}
