#pragma once
#include "flash_fwd_kernel.cuh"
#include "config.cuh"
namespace gpt2cuda {
namespace kernel {

    template<AttentionType Attention ,int Hc, int Br = 64 , int Bc = 64 >
    void AttentionForwardCUDA(float *outputs, float * logsumexp ,float const *inputs, int B, int T, int C3, int NH,cudaStream_t stream) {

        assert(C3 % 3 == 0);
        int C = C3 / 3;
        assert(C % NH == 0);
        assert(C / NH  == Hc);
        assert(T / Br  > 0);
        assert(T % Br == 0);
        assert(T / Bc  > 0);
        assert(T % Bc == 0);
        CUTE_STATIC_ASSERT_V(bool_constant<Hc == 32 || Hc == 64>(),"Hc is not supported");
        
        

        auto L_Q = make_layout(make_shape(B, NH, T,Int<Hc>{}), make_stride(T * C3, Int<Hc>{}, C3, Int<1>{}));
        auto L_K = make_layout(make_shape(B, NH, T,Int<Hc>{}), make_stride(T * C3, Int<Hc>{}, C3, Int<1>{}));
        auto L_V = make_layout(make_shape(B, NH, T,Int<Hc>{}), make_stride(T * C3, Int<Hc>{}, C3, Int<1>{}));
        auto L_O = make_layout(make_shape(B, NH, T,Int<Hc>{}), make_stride(T * C, Int<Hc>{}, C, Int<1>{}));
        auto L_logsumexp = make_layout(make_shape(B,NH,T), make_stride(T * NH, T ,Int<1>{}));

        TiledCopy copyQ = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cutlass::uint128_t>, float>{},
                Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                Layout<Shape<_1, _4>>{});

        TiledCopy copyK = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cutlass::uint128_t>, float>{},
                Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                Layout<Shape<_1, _4>>{});

        TiledCopy copyV = make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, cute::half_t>{},
                Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                Layout<Shape<_1, _4>>{});

        TiledCopy copyO = make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint128_t>, float>{},
                Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                Layout<Shape<_1, _4>>{});

        TiledMMA mmaS = make_tiled_mma(SM80_16x8x8_F32TF32TF32F32_TN{}, Layout<Shape<_4, _1, _1>>{}, Tile<Int<Br>, Int<Bc>, _8>{});

        TiledMMA mmaO = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<_4, _1, _1>>{}, Tile<Int<Br>, Int<Hc>, _16>{});

        // print(copyQ);
        // print(copyK);
        // print(mmaS);

        // print_latex(copyQ);
        // print_latex(copyK);
        // print_latex(mmaS);
        // print_latex(mmaO);
        // mma base
        // print_latex(make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},Layout<Shape<_1,_1,_1>>{},Tile<_16,_8,_16>{}));

        auto kernel_fptr = AttentionForwardKernel<float, decltype(L_Q), decltype(copyQ),
                float, decltype(L_K), decltype(copyK),
                float, decltype(L_V), decltype(copyV),
                float, decltype(L_O), decltype(copyO),
                float, decltype(L_logsumexp),
                decltype(mmaS), decltype(mmaO),
                Br,Bc,Hc,Attention>;


        
        dim3 dimGrid(B,NH);
        kernel_fptr<<<dimGrid, 128,0,stream>>>(inputs, L_Q, copyQ, inputs + C, L_K, copyK, inputs + 2 * C, L_V, copyV, outputs, L_O, copyO, logsumexp,L_logsumexp,mmaS, mmaO);
    }




}
} //namespace kernel
