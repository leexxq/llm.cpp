#include "cuda/error.cuh"
#include "flash_bwd_kernel.cuh"
#include "cutlass/util/device_memory.h"
#include "config.cuh"
#include <stdexcept>

namespace gpt2cuda {
namespace kernel {
    
    template <class LQKV,class LO,class LLD>
	struct BwdLayouts{
		using LayoutQ = LQKV;
		using LayoutK = LQKV;
		using LayoutV = LQKV;
		using LayoutO = LO;
		using LayoutdO = LayoutO;
		using LayoutL = LLD;
		using LayoutD = LLD;
		using LayoutdQ = LQKV;
		using LayoutdK	= LQKV;
		using LayoutdV = LQKV;
	};
    template<AttentionType Attention ,int Hc>
    void AttentionBackwardCUDA(float *d_inputs, float * D , float const *d_outputs, float const* outputs , float const *inputs,float const* logsumexp, int B, int T, int C3, int NH,cudaStream_t stream) {

        using Config = FlashBwdConfigFp32<Hc,Attention>;
        int C = C3 / 3;
        if(!(C3 % 3 == 0 && C % NH == 0 && C / NH  == Hc)){
            throw std::invalid_argument("input C is mismatch");

        }
        if(!(T / Config::kBr  > 0 && T % Config::kBr == 0 &&T / Config::kBc  > 0 &&T % Config::kBc == 0)){
            throw std::invalid_argument("input T is mismatch");
        }
        CUTE_STATIC_ASSERT_V(bool_constant<Hc == 32 || Hc == 64>(),"Hc is not supported");


        auto L_Q= make_layout(make_shape(int{B}, int{NH}, int{T},Int<Hc>{}), make_stride(int{T} * int{C3}, Int<Hc>{}, int{C3}, Int<1>{}));
        
        auto L_K = L_Q ,L_V = L_Q; 

        auto L_O = make_layout(make_shape(int{B}, int{NH}, int{T},Int<Hc>{}), make_stride(int{T} * int{C}, Int<Hc>{}, C, Int<1>{}));
        auto L_dO =L_O; 
        auto L_L = make_layout(make_shape(int{B}, int{NH}, int{T}), make_stride(int{T} * int{NH}, int{T} ,Int<1>{}));
        auto L_D = L_L; 


        dim3 dimGrid(B,NH);
        auto dQ = d_inputs ;
        auto dK = d_inputs + C;
        auto dV = d_inputs + 2*C;
        auto Q = inputs;
        auto K = inputs + C;
        auto V = inputs + 2 * C;
        auto O = d_outputs;
        auto L = logsumexp;


        


        auto kernel_fptr = AttentionBackwardKernel<Config,BwdLayouts<decltype(L_Q), decltype(L_O),decltype(L_L)>>;
        // std::cout << "bwd using smem size : " << Config::smem_size/ 1024 << "kb" << std::endl;
        CUDA_CHECK(cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, Config::smem_size));
        CUDA_CHECK(cudaFuncSetAttribute(kernel_fptr,cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        kernel_fptr<<<dimGrid,Config::kthreads,Config::smem_size,stream>>>(
                                    Q,L_Q,
                                    K,L_K,
                                    V,L_V,
                                    outputs,L_O,
                                    d_outputs,L_dO,
                                    L,L_L,
                                    D,L_D,
                                    dQ,L_Q,
                                    dK,L_K,
                                    dV,L_V);
        CUDA_CHECK_LAST();
    }
}
}