#include "cuda/error.cuh"
#include "flash_bwd_kernel.cuh"
#include "cutlass/util/device_memory.h"
#include "config.cuh"

namespace gpt2cuda {
namespace kernel {
    
    template <class LQKV,class LO,class LLD>
	struct Layouts{
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
    template<AttentionType Attention ,int Hc, int Br = 64 , int Bc = 64 >
    void AttentionBackwardCUDA(float *d_inputs, float const *d_outputs, float const* outputs , float const *inputs,float const* logsumexp, int B, int T, int C3, int NH) {

        assert(C3 % 3 == 0);
        int C = C3 / 3;
        assert(C % NH == 0);
        assert(C / NH  == Hc);
        assert(T / Br  > 0);
        assert(T % Br == 0);
        assert(T / Bc  > 0);
        assert(T % Bc == 0);
        CUTE_STATIC_ASSERT_V(bool_constant<Hc == 32 || Hc == 64>(),"Hc is not supported");

        auto D = cutlass::device_memory::allocation<float>(B*NH*T);

        auto L_Q= make_layout(make_shape(int{B}, int{NH}, int{T},Int<Hc>{}), make_stride(int{T} * int{C3}, Int<Hc>{}, int{C3}, Int<1>{}));
        
        auto L_K = L_Q ,L_V = L_Q; 

        auto L_O = make_layout(make_shape(int{B}, int{NH}, int{T},Int<Hc>{}), make_stride(int{T} * int{C}, Int<Hc>{}, C, Int<1>{}));
        auto L_dO =L_O; 
        auto L_L = make_layout(make_shape(int{B}, int{NH}, int{T}), make_stride(int{T} * int{NH}, int{T} ,Int<1>{}));
        auto L_D = L_L; 

        // decltype(L_Q) L_test;
        // print_layout(L_test);
        // TiledCopy copyQ = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cutlass::uint128_t>, float>{},
        // 		Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        // 		Layout<Shape<_1, _4>>{});

        // TiledCopy copyV = make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, cute::half_t>{},
        // 		Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        // 		Layout<Shape<_1, _4>>{});
        


        // {
        // 	TiledMMA mmaS = make_tiled_mma(SM80_16x8x8_F32F16F16F32_TN{}, Layout<Shape<_1, _4, _1>>{}, Tile<Int<16>, Int<32>, _8>{});
        // 	print_latex(mmaS);
        // }

        // {
        // 	TiledMMA mmadV = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<_1, _4, _1>>{}, Tile<Int<16>, Int<32>, _16>{});
        // 	print_latex(mmadV);
        // }



        dim3 dimGrid(B,NH);
        auto dQ = d_inputs ;
        auto dK = d_inputs + C;
        auto dV = d_inputs + 2*C;
        auto Q = inputs;
        auto K = inputs + C;
        auto V = inputs + 2 * C;
        auto O = d_outputs;
        auto L = logsumexp;


        
        using Config = FlashBwdConfigFp32<Hc,Attention>;



        auto kernel_fptr = AttentionBackwardKernel<Config,Layouts<decltype(L_Q), decltype(L_O),decltype(L_L)>>;
        // std::cout << "using smem size : " << Config::smem_size/ 1024 << "kb" << std::endl;
        CUDA_CHECK(cudaFuncSetAttribute(kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, Config::smem_size));
        CUDA_CHECK(cudaFuncSetAttribute(kernel_fptr,cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        kernel_fptr<<<dimGrid,Config::kthreads,Config::smem_size>>>(
                                    Q,L_Q,
                                    K,L_K,
                                    V,L_V,
                                    outputs,L_O,
                                    d_outputs,L_dO,
                                    L,L_L,
                                    D.get(),L_D,
                                    dQ,L_Q,
                                    dK,L_K,
                                    dV,L_V);
        CUDA_CHECK_LAST();
    }
}
}