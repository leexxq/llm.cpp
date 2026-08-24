#pragma once
#include "cuda/error.cuh"
#include "cute/util/print_latex.hpp"
#include "flash_fwd_kernel.cuh"
#include "config.cuh"
#include <stdexcept>
namespace gpt2cuda {
namespace kernel {
    template <class LQKV,class LO,class LL>
        struct FwdLayouts{
                using LayoutQ = LQKV;
                using LayoutK = LQKV;
                using LayoutV = LQKV;
                using LayoutO = LO;
                using LayoutL = LL;
        };
    template<AttentionType Attention ,int Hc>
    void AttentionForwardCUDA(float *outputs, float * logsumexp ,float const *inputs, int B, int T, int C3, int NH,cudaStream_t stream) {
        using Config = FlashFwdConfigFp32<Hc,Attention>;

        if(C3 % 3 != 0){
            throw std::invalid_argument("inputs C mismatch");
        }
        int C = C3 / 3;
        if(!(C % NH == 0 ||C / NH  == Hc)){
            throw std::invalid_argument("inputs C mismatch");
        }
        if(!(T / Config::kBr  > 0 && T % Config::kBr == 0 && T / Config::kBc  > 0 && T % Config::kBc == 0)){
            throw std::invalid_argument("inputs T mismatch");

        }        

        CUTE_STATIC_ASSERT_V(bool_constant<Hc == 32 || Hc == 64>(),"Hc is not supported");
        
        

        auto L_Q = make_layout(make_shape(B, NH, T,Int<Hc>{}), make_stride(T * C3, Int<Hc>{}, C3, Int<1>{}));
        auto L_K = make_layout(make_shape(B, NH, T,Int<Hc>{}), make_stride(T * C3, Int<Hc>{}, C3, Int<1>{}));
        auto L_V = make_layout(make_shape(B, NH, T,Int<Hc>{}), make_stride(T * C3, Int<Hc>{}, C3, Int<1>{}));
        auto L_O = make_layout(make_shape(B, NH, T,Int<Hc>{}), make_stride(T * C, Int<Hc>{}, C, Int<1>{}));
        auto L_logsumexp = make_layout(make_shape(B,NH,T), make_stride(T * NH, T ,Int<1>{}));


        // print(copyQ);
        // print(copyK);
        // print(mmaS);

        // print_latex(copyQ);
        // print_latex(copyK);
        // print_latex(mmaS);
        // print_latex(mmaO);
        // mma base
        // print_latex(make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},Layout<Shape<_1,_1,_1>>{},Tile<_16,_8,_16>{}));

        // print_latex(typename Config::SQ_Swz{});
        // print_latex(typename Config::SK_Swz{});
        // print_latex(typename Config::SV_Swz{});
        

        auto kernel_fptr = AttentionForwardKernel<Config,FwdLayouts<decltype(L_Q), decltype(L_O), decltype(L_logsumexp)>>;

        dim3 dimGrid(B,NH);
        // std::cout << "fwd using smem size : " << Config::smem_size/ 1024 << "kb" << std::endl;
        CUDA_CHECK(cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, Config::smem_size));
        CUDA_CHECK(cudaFuncSetAttribute(kernel_fptr,cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        kernel_fptr<<<dimGrid, Config::kthreads,Config::smem_size,stream>>>(inputs, L_Q, inputs + C, L_K,  inputs + 2 * C, L_V, outputs, L_O,  logsumexp,L_logsumexp);
        CUDA_CHECK_LAST();
    }




}
} //namespace kernel
