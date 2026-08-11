#include "error.cuh"
#include "cutlass/util/exceptions.h"
#include "encoder.cuh"
#include "cutlass/util/device_memory.h"
#include <cstdio>
namespace gpt2cuda {
    namespace kernel {

        __global__ void EncoderForwardKernel(float * outputs,int const * inputs ,float  const * wte,float const*  wpe,int B,int T,int C){
            const int idx = threadIdx.x + blockDim.x * blockIdx.x;
            const int row = (idx/C); //(0,B*T-1)
            
            if(idx < B*T*C){
                outputs[idx] = wte[inputs[row]*C + idx%C] + wpe[(row % T)*C + idx%C];
            }
        }

        __global__ void EncoderBackwardKernel(float* d_wte,float * d_wpe, float const*  d_outputs,int const * inputs,int B,int T,int C){
            const int idx = threadIdx.x + blockDim.x * blockIdx.x;
            const int row = idx / C;//(0,B*T-1)
            if(idx < B*T*C){
                atomicAdd(&d_wte[inputs[row]*C + idx %C], d_outputs[idx]);
                atomicAdd(&d_wpe[(row % T)*C + idx%C],d_outputs[idx]);
            }
        }

        template <int threads=256>
        void EncoderForwardCUDA(float * outputs,int const * inputs ,float  const * wte,float const*  wpe,int B,int T,int C){
            EncoderForwardKernel<<<(B*T*C + threads - 1)/ threads,threads>>>(outputs, inputs, wte, wpe, B, T, C);
            CUDA_CHECK_LAST();
        }


        template <int threads=256>
        void EncoderBackwardCUDA(float* d_wte,float * d_wpe, float const*  d_outputs,int const * inputs,int B,int T,int C){

            EncoderBackwardKernel<<<(B*T*C + threads - 1)/ threads,threads>>>(d_wte,d_wpe,d_outputs,inputs ,B,T,C);
            CUDA_CHECK_LAST();
        }
        
    }

    using DAllocf = cutlass::device_memory::allocation<float>;
    using DAlloci = cutlass::device_memory::allocation<int>;
    void BatchEncoderForward(float * outputs,int const * inputs ,float  const * wte,float const*  wpe,int B,int T,int C,int Vp,int MaxT){
        DAlloci inputs_d(B*T);
        DAllocf outputs_d(B*T*C);
        DAllocf wte_d(Vp*C);
        DAllocf wpe_d(MaxT*C);

        inputs_d.copy_from_host(inputs);
        wte_d.copy_from_host(wte);
        wpe_d.copy_from_host(wpe);


        kernel::EncoderForwardCUDA(outputs_d.get(), inputs_d.get(),wte_d.get(),wpe_d.get(), B, T, C);

        try {
            outputs_d.copy_to_host(outputs);
        } catch (cutlass::cuda_exception& e) {
            std::cerr<< e << std::endl;
        }

    }

    using DAlloc = cutlass::device_memory::allocation<float>;
    void BatchEncoderBackward(float* d_wte,float * d_wpe, float const*  d_outputs,int const * inputs,int B,int T,int C,int Vp,int MaxT){
        DAllocf d_wte_d(Vp*C);
        DAllocf d_wpe_d(MaxT*C);

        DAlloci inputs_d(B*T);
        DAllocf d_outputs_d(B*T*C);

        d_wte_d.copy_from_host(d_wte);
        d_wpe_d.copy_from_host(d_wpe);
        d_outputs_d.copy_from_host(d_outputs);
        inputs_d.copy_from_host(inputs);

        kernel::EncoderBackwardCUDA(d_wte_d.get(), d_wpe_d.get(), d_outputs_d.get(), inputs_d.get(), B, T, C);

        try {
            d_wte_d.copy_to_host(d_wte);
            d_wpe_d.copy_to_host(d_wpe);
        } catch (cutlass::cuda_exception& e) {
            std::cerr<< e << std::endl;
        }

    }
}