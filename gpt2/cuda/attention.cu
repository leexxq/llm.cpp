#include "attention.cuh"
#include "cute/tensor.hpp"
#include "cute/util/debug.hpp"
#include "cute/util/print_tensor.hpp"
#include "cutlass/numeric_conversion.h"
#include "cutlass/util/device_memory.h"
#include <cassert>
namespace gpt2cuda {
namespace kernel {
    using namespace cute;

    template<class To_type ,class Engine, class Layout>
    __forceinline__ __device__ auto convert_type(Tensor<Engine,Layout> const & tensor){

        using From_type = typename Engine::value_type;

        Tensor target = make_fragment_like<To_type>(tensor.layout());

        auto convert_op = cutlass::NumericConverter<To_type, From_type>{};

        CUTE_UNROLL
        for(int i =0 ; i < size(tensor); ++i){
            target(i) = convert_op(tensor(i));
        }
        return target;

    }

    template< class O_tensor  , class Sum_tensor>
    __forceinline__ __device__  auto normalize_scale_o(O_tensor row_tOrO , Sum_tensor tsum_frag){
        CUTE_UNROLL
        for(int r =0 ; r < size<1>(row_tOrO); ++r){
            for(int c = 0 ; c < size<0>(row_tOrO); ++c){
                row_tOrO(c,r) =  row_tOrO(c,r) / tsum_frag(r);
            }
        }

    }


    template<bool Is_first,class S_tensor , class O_tensor ,class Max_tensor , class Sum_tensor>
    __forceinline__ __device__ auto scaled_softmax(S_tensor row_tSrS , O_tensor row_tOrO, Max_tensor tmaxs_frag, Sum_tensor tsums_frag){

        constexpr float klog_2e = 1.442695;
        constexpr float kscale = 0.176777;  // 1 / sqrt(32);
        constexpr float klog2_scale = klog_2e * kscale;

        CUTE_STATIC_ASSERT_V(size<1>(row_tOrO) == size<1>(row_tSrS));


        // reduce sum 
        //max
        CUTE_UNROLL
        for(int r = 0; r < size<1>(row_tSrS);++r){
            float tmax_frag_old ;
            if constexpr (!Is_first) {
                tmax_frag_old = tmaxs_frag(r);
            }

            tmaxs_frag(r) = row_tSrS(0,r);

            CUTE_UNROLL
            for(int c = 1; c < size<0>(row_tSrS); ++c){
                // per thread per sum
                tmaxs_frag(r) = max(tmaxs_frag(r),row_tSrS(c,r));
            }

            // quad reduce max
            CUTE_UNROLL
            for(int off = 1; off < 4; off <<= 1){
                tmaxs_frag(r) = max(__shfl_xor_sync(0xFFFFFFFFU,tmaxs_frag(r),off),tmaxs_frag(r));
            }

            
            float tmax_frag_new;
            if constexpr (Is_first){
                tmax_frag_new = tmaxs_frag(r);
            }else{
                tmax_frag_new = max(tmaxs_frag(r),tmax_frag_old);
            }


            // scale_softmax : softmax(QK^T);
            CUTE_UNROLL
            for(int c = 0; c < size<0>(row_tSrS); ++c){
                row_tSrS(c,r) = exp2f(klog2_scale * (row_tSrS(c,r) - tmax_frag_new));
            }

            // per thread sum 
            if constexpr(Is_first){
                tsums_frag(r) = row_tSrS(0,r);
            }else{
                float correction = exp2f(klog2_scale * (tmax_frag_old - tmax_frag_new));
                tsums_frag(r) = correction * tsums_frag(r)  + row_tSrS(0,r);
                CUTE_UNROLL
                for(int c = 0 ; c < size<0>(row_tOrO); ++c){
                    row_tOrO(c,r) =  correction * row_tOrO(c,r);
                }
            }

            CUTE_UNROLL
            for(int c = 1; c < size<0>(row_tSrS); ++c){
                // per thread sum
                tsums_frag(r) += row_tSrS(c,r);
            }
            // quad reduce sum
            for(int off = 1; off < 4; off <<= 1){
                tsums_frag(r) +=__shfl_xor_sync(0xFFFFFFFFU,tsums_frag(r),off);
            }

            tmaxs_frag(r) = tmax_frag_new;
        }

        // if(thread0()){
        //     print("per thread : tmax_frag_");
        //     print_tensor(tmax_frag);
        // }


        // if(thread0()){
        //     print("reduce : tmax_frag_");
        //     print_tensor(tmax_frag);
        // }




        // if(thread0()){
        //     print("scale_softmax : row_tSrs_");
        //     print_tensor(row_tSrS);
        // }
        

 

        // if(thread0()){
        //     print("per thread tsum_frag_");
        //     print_tensor(tsum_frag);
        // }

 

        // if(thread0()){
        //     print("reduce : tsum_frag_");
        //     print_tensor(tsums_frag);
        // }
        


    }


    

    // Here, "row" means that I place a row of data owned by a thread at the first index, 
    // while the second index corresponds to other rows.
    template<class C_tensor>
    __forceinline__ __device__ auto convert_C_to_row(C_tensor ctensor){
        // retile the register file layout (due per thread ownning multi row datas, we each calculate them.)

        auto row_tiler = make_layout(
            make_shape(size<0,0>(ctensor),Int<1>{},size<2>(ctensor)),
            make_stride(Int<1>{},Int<0>{},Int<size<0>(ctensor) * size<1>(ctensor)>{})
        );
        return zipped_divide(ctensor,row_tiler); //(nums_pre_row,rows)
    }





    // flash attention v2 ，see also https://arxiv.org/pdf/2307.08691
    // visit https://blog.echen.io/p/flashattention-2-in-cute-from-scratch
    template<class TQ , class LayoutQ,class TiledCopyQ,
             class TK , class LayoutK,class TiledCopyK,
             class TV , class LayoutV,class TiledCopyV,
             class TO , class LayoutO,class TiledCopyO,
             class TiledMmaS,class TiledMmaO,
             int Br = 64 ,int Bc = 64>
    __global__ void  AttentionForwardKernel(TQ const * Q , LayoutQ L_Q , TiledCopyQ copy_Q , 
                                TK const * K , LayoutK L_K , TiledCopyK copy_K , 
                                TV const * V , LayoutV L_V , TiledCopyV copy_V , 
                                TO const * O , LayoutO L_O , TiledCopyO copy_O ,
                                TiledMmaS mmaS , TiledMmaO mmaO){

        
        auto Hc = Int<32>{};

        __shared__ float shared_memQ[Br*Hc];

        __shared__ float shared_memK[Bc*Hc];

        __shared__ cute::half_t shared_memV[Bc * Hc];


        auto block_Q = make_tensor(Q,L_Q)(blockIdx.x,blockIdx.y,_,_);//(T,Hc)
        auto block_K = make_tensor(K,L_K)(blockIdx.x,blockIdx.y,_,_);//(T,Hc)
        auto block_V = make_tensor(V,L_V)(blockIdx.x,blockIdx.y,_,_);//(T,Hc)
        auto block_O = make_tensor(O,L_O)(blockIdx.x,blockIdx.y,_,_);//(T,Hc)

        // auto Tr =  size<0>(block_Q) / Br;
        // auto Tc =  size<0>(block_K) / Bc;
        auto Tr =  1;
        auto Tc =  1;
        
        for(int tr = 0 ; tr < Tr; ++tr){

            Tensor gQ = local_tile(block_Q,make_tile(Int<Br>{},Int<Hc>{}),make_coord(tr,0));
            Tensor gO = local_tile(block_O,make_tile(Int<Br>{},Int<Hc>{}),make_coord(tr,0));
            //alloc fragment for S
            Tensor tSrS = partition_fragment_C(mmaS, make_shape(Int<Br>{},Int<Bc>{}));
            clear(tSrS);

            // retile the register file layout (due per thread ownning multi row datas, we each calculate them.)

        
            Tensor row_tSrS = convert_C_to_row(tSrS); //(nums_pre_row,rows)

            // alloc register for sum op and max op;
            Tensor tsum_frag = make_tensor<float>(Int<size<1>(row_tSrS)>{});
            Tensor tmax_frag = make_tensor<float>(Int<size<1>(row_tSrS)>{});

            //alloc fragment for O
            Tensor tOrO = partition_fragment_C(mmaO, make_shape(Int<Br>{},Int<Hc>{}));
            clear(tOrO);

            // retile the register file layout (due per thread ownning multi row datas, we each calculate them.)
            Tensor row_tOrO = convert_C_to_row(tOrO); //(nums_pre_row,rows)

            for(int tc = 0 ; tc < Tc; ++tc){
                Tensor gK = local_tile(block_K,make_tile(Int<Bc>{},Int<Hc>{}),make_coord(tc,0));
                Tensor gV = local_tile(block_V,make_tile(Int<Bc>{},Int<Hc>{}),make_coord(tc,0));


                // print_tensor(thr_tile_S);


                //Qmem
                Layout sQ_layout = make_layout(make_shape(Int<Br>{},Int<Hc>{}),LayoutRight{});
                Tensor sQ = make_tensor(make_smem_ptr(shared_memQ),sQ_layout); //(T,Hc,k) 

                //Kmem
                Layout sK_layout = make_layout(make_shape(Int<Bc>{},Int<Hc>{}),LayoutRight{});
                Tensor sK = make_tensor(make_smem_ptr(shared_memK),sK_layout);//(T,Hc,k)

                

                //Qcopy
                ThrCopy thr_copy_Q = copy_Q.get_slice(threadIdx.x);
                Tensor tQgQ = thr_copy_Q.partition_S(gQ); //(CPY,CPY_Br,CPY_Hc)
                Tensor tQsQ = thr_copy_Q.partition_D(sQ); //(CPY,CPY_T,CPY_Hc)

                //Kcopy
                ThrCopy thr_copy_K = copy_K.get_slice(threadIdx.x);
                Tensor tKgK = thr_copy_K.partition_S(gK); //(CPY,CPY_Bc,CPY_Hc)
                Tensor tKsK = thr_copy_K.partition_D(sK);//(CPY,CPY_T,CPY_Hc)

                // async copy q and k
                if(tc == 0){
                    copy(copy_Q,tQgQ,tQsQ);
                }
                copy(copy_K,tKgK,tKsK);
                cp_async_fence();
                // async copy v




                //wait q and k
                cp_async_wait<0>();
                __syncthreads();



                ThrMMA thr_mmaS = mmaS.get_slice(threadIdx.x);

                Tensor tSrQ = thr_mmaS.partition_fragment_A(sQ); // (MMA, MMA_Br,MMA_Hc)
                Tensor tSrK = thr_mmaS.partition_fragment_B(sK); // (MMA, MMA_Br,MMA_Hc)


                // Copy_Atom<UniversalCopy<float>, float> s2r_atom_Q;
                // Copy_Atom<UniversalCopy<float>, float> s2r_atom_K;
                
                Copy_Atom<SM75_U32x4_LDSM_N,float> s2r_atom_Q;
                Copy_Atom<SM75_U32x4_LDSM_N,float> s2r_atom_K;
                TiledCopy s2r_copy_Q = make_tiled_copy_A(s2r_atom_Q,mmaS);
                ThrCopy s2r_thr_copy_Q = s2r_copy_Q.get_slice(threadIdx.x);

                Tensor tXsQ = s2r_thr_copy_Q.partition_S(sQ);
                Tensor tXrQ = s2r_thr_copy_Q.retile_D(tSrQ);

                TiledCopy s2r_copy_K = make_tiled_copy_B(s2r_atom_K,mmaS);
                ThrCopy s2r_thr_copy_K = s2r_copy_K.get_slice(threadIdx.x);

                Tensor tXsK = s2r_thr_copy_K.partition_S(sK); //(CPY,MMA_T,MMA_Hc)
                Tensor tXrK = s2r_thr_copy_K.retile_D(tSrK); 




                copy(s2r_atom_Q,tXsQ,tXrQ);
                copy(s2r_atom_K,tXsK,tXrK);


                if(thread0()){
                    print("tSrQ:");
                    print_tensor(tSrQ);
                    print("tSrK:");
                    print_tensor(tSrK);
                }
                

                // S = QK^T
                for(int k =0 ; k < size<2>(tSrQ); ++k){
                    gemm(mmaS,tSrQ(_,_,k),tSrK(_,_,k),tSrS);
                }
                
                if(thread0()){
                    // print_tensor(sQ);
                    // print("tSrS:");
                    // print_tensor(tSrS);
                    // print("flat tSrS:");
                    // for(int i =0 ; i < tSrS.size(); ++i){
                    //     print(tSrS(i));
                    //     print(" ");
                    // }
                    // print("\n");

                    print("row_tSrS:");
                    print_tensor(row_tSrS);

                }

                // online softmax 
                if(tc == 0){
                    scaled_softmax<true>(row_tSrS,row_tOrO,tmax_frag, tsum_frag);
                }else{
                    scaled_softmax<false>(row_tSrS,row_tOrO,tmax_frag, tsum_frag);
                }

                //now we get per row from formula softmax(QK^T)

                // V smem (swizzle)
                Layout sV_layout = make_layout(make_shape(Int<Bc>{},Int<Hc>{}),LayoutRight{});
                // Tensor sV_swz = make_tensor(make_smem_ptr(shared_memV),composition(Swizzle<5,0,5>{},sV_layout));
                Tensor sV = make_tensor(make_smem_ptr(shared_memV),sV_layout);
                ThrCopy thr_copy_V = copy_V.get_slice(threadIdx.x);

                Tensor tVgV = thr_copy_V.partition_S(gV);


                // copy global to register and convert fp32 to fp16.
                // finally, copy to smem
                {
                    Tensor tVrV = make_fragment_like<float>(tVgV);
                    Tensor tXrV = thr_copy_V.retile_D(tVrV);
                    copy(copy_V,tVgV,tXrV);

                    if(thread0()){
                        print("gV:");
                        print_tensor(gV);
                        print("tVgV:");
                        print_tensor(tVgV);
                        print("tVrV:");
                        print_tensor(tVrV);
                    }

                    //convert to half type
                    Tensor tVrV_fp16 =  convert_type<cute::half_t>(tVrV);

                    if(thread0()){
                        print("tVrV fp16:");
                        print_tensor(tVrV_fp16);
                    }

                    Tensor tVsV = thr_copy_V.partition_D(sV);
                    copy(copy_V,tXrV,tVsV);
                }



                if(thread0()){
                    print("sV:");
                    print_tensor(sV);
                }

                Layout sVT_layout = make_layout(make_shape(Int<Hc>{},Int<Bc>{}),LayoutLeft{});
                Tensor sVT = make_tensor(make_smem_ptr(shared_memV),sVT_layout);
                
                Copy_Atom<SM75_U16x8_LDSM_T,cute::half_t> s2r_atom_V;

                TiledCopy s2r_copy_VT  = make_tiled_copy_B(s2r_atom_V, mmaO);
                ThrCopy s2r_thr_copy_VT = s2r_copy_VT.get_slice(threadIdx.x);

                ThrMMA thr_mmaO = mmaO.get_slice(threadIdx.x);

                Tensor tVrVT = thr_mmaO.partition_fragment_B(sVT); // (MMA, MMA_Bc,MMA_Hc)
                
                Tensor tXsVT = s2r_thr_copy_VT.partition_S(sVT);
                Tensor tXrVT = s2r_thr_copy_VT.retile_D(tVrVT);

                copy(s2r_atom_V,tXsVT,tXrVT);

                if(thread0()){
                    print("sVT:");
                    print_tensor(sVT);
                    print("tVrVT:");
                    print_tensor(tVrVT);
                }
                Tensor tSrS_fp16 = convert_type<cute::half_t>(tSrS);
                if(thread0()){
                    print("tSrS:");
                    print_tensor(tSrS);
                    print("tSrS_fp16:");
                    print_tensor(tSrS_fp16);
                }
                
                // convert s frag to a frag for o 
                auto retile_shape = make_shape(make_shape(size<0,0>(tSrS_fp16),size<0,1>(tSrS_fp16),_2{}),size<1>(tSrS_fp16),size<2>(tSrS_fp16) / _2{});
                auto retile_stride = make_stride(make_stride(stride<0,0>(tSrS_fp16),stride<0,1>(tSrS_fp16),stride<2>(tSrS_fp16)),stride<1>(tSrS_fp16),stride<2>(tSrS_fp16) * _2{});

                Tensor tSrS_fp16_2_A_frag = make_tensor(tSrS_fp16.data(),make_layout(retile_shape,retile_stride));

                if(thread0()){
                    print("tSrS_fp16_2_A_frag:");
                    print_tensor(tSrS_fp16_2_A_frag);
                }

                // we will calculate SV
                // due O = SV
                for(int k =0 ; k < size<2>(tVrVT); ++k){
                    gemm(mmaS,tSrS_fp16_2_A_frag(_,_,k),tVrVT(_,_,k),tOrO);
                }

                if(thread0()){
                    print("tOrO:");
                    print_tensor(tOrO);
                }

            }
            // scale o ... 
            normalize_scale_o(row_tOrO,tsum_frag);
            

            // copy O to Q for coalease write
            Layout sO_layout = make_layout(make_shape(Int<Br>{},Int<Hc>{}),LayoutRight{});
            Tensor sO = make_tensor(make_smem_ptr(shared_memQ),sO_layout);

            using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>,float>;

            TiledCopy r2s_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, mmaO);

            ThrCopy smem_thr_copy_O = r2s_tiled_copy_O.get_slice(threadIdx.x);

            Tensor copy_trO = smem_thr_copy_O.retile_S(tOrO);
            Tensor copy_tsO = smem_thr_copy_O.partition_D(sO);


            __syncthreads();
            copy(r2s_tiled_copy_O,copy_trO,copy_tsO);

            ThrCopy s2g_thr_tiled_copy = copy_O.get_slice(threadIdx.x);
            Tensor copy_tOsO = s2g_thr_tiled_copy.partition_S(sO);
            Tensor copy_tOgO = s2g_thr_tiled_copy.partition_D(gO);

            Tensor copy_tOrO = make_fragment_like(copy_tOgO);


            // // smem -> registers
            // copy(copy_O,copy_tOsO,copy_tOrO);

            // // registers -> gmem
            // copy(copy_O,copy_tOrO,copy_tOgO);

            

            

        }
    }

    void AttentionForwardCUDA(float * outputs , float const * inputs,int B,int T ,int C3,int NH){
        using namespace cute;

        assert(C3 % 3 ==0);
        int C = C3 / 3;
        assert(C % NH ==0);
        int Hc = C / NH; 
        assert(Hc == 32);

        auto L_Q = make_layout(make_shape(B,NH,T,_32{}),make_stride(T*C3,_32{},C3,Int<1>{}));
        auto L_K = make_layout(make_shape(B,NH,T,_32{}),make_stride(T*C3,_32{},C3,Int<1>{}));
        auto L_V = make_layout(make_shape(B,NH,T,_32{}),make_stride(T*C3,_32{},C3,Int<1>{}));
        auto L_O = make_layout(make_shape(B,NH,T,_32{}),make_stride(T*C,_32{},C,Int<1>{}));


        TiledCopy copyQ = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cutlass::uint128_t>,float>{},
                                        Layout<Shape<_16,_8>,Stride<_8,_1>>{},
                                        Layout<Shape<_1,_4>>{});

        TiledCopy copyK = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cutlass::uint128_t>,float>{},
                                        Layout<Shape<_16,_8>,Stride<_8,_1>>{},
                                        Layout<Shape<_1,_4>>{});

        TiledCopy copyV = make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>,cute::half_t>{},
                                        Layout<Shape<_16,_8>,Stride<_8,_1>>{},
                                        Layout<Shape<_1,_4>>{});

        TiledCopy copyO = make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint128_t>,float>{},
                                    Layout<Shape<_16,_8>,Stride<_8,_1>>{},
                                    Layout<Shape<_1,_4>>{});


        // TiledMMA mmaS = make_tiled_mma(SM80_16x8x8_F32TF32TF32F32_TN{},Layout<Shape<Shape<_4,_4>,_1,_1>,Stride<Stride<_0,_1>,_1,_1>>{},Tile<_256,_1/*Layout<Shape<_2,_4,_4>,Stride<_1,_8,_2>>*/,_8>{});
        // TiledMMA mmaS = make_tiled_mma(SM80_16x8x8_F32TF32TF32F32_TN{},Layout<Shape<_4,_1,_1>>{},Tile<_64,Layout<Shape<_2,_4,_4>,Stride<_1,_8,_2>>,_8>{});
        TiledMMA mmaS = make_tiled_mma(SM80_16x8x8_F32TF32TF32F32_TN{},Layout<Shape<_4,_1,_1>>{},Tile<_64,_64,_8>{});

        TiledMMA mmaO = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},Layout<Shape<_4,_1,_1>>{},Tile<_64,_32,_64>{});

        

        // print(copyQ);
        // print(copyK);
        // print(mmaS);


        // print_latex(copyQ);
        // print_latex(copyK);
        // print_latex(mmaS);
        // print_latex(mmaO);
        // mma base
        // print_latex(make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},Layout<Shape<_1,_1,_1>>{},Tile<_16,_8,_16>{}));

        auto kernel_fptr = AttentionForwardKernel<float , decltype(L_Q) , decltype(copyQ),
                                                  float , decltype(L_K) , decltype(copyK),
                                                  float , decltype(L_V) , decltype(copyV),
                                                  float , decltype(L_O) , decltype(copyO),
                                                  decltype(mmaS),decltype(mmaO)
                                                  >;
        kernel_fptr<<<1,128>>>(inputs, L_Q, copyQ, inputs + C, L_K,  copyK, inputs+2*C,L_V,copyV, outputs,L_O,copyO,mmaS, mmaO);

    }


}
using DAlloc = cutlass::device_memory::allocation<float>;


void BatchAttentionForward(float * outputs , float const * inputs,int B,int T ,int C3,int NH){


    using namespace cute;
    assert(C3 % 3 ==0);
    auto C = C3 / 3;


    DAlloc outputs_d(B*T*C);
    DAlloc inputs_d(B*T*C3);

    outputs_d.copy_from_host(outputs);
    inputs_d.copy_from_host(inputs);

    kernel::AttentionForwardCUDA(outputs_d.get(), inputs_d.get(), B, T, C3, NH);

    outputs_d.copy_to_host(outputs);

 
}

void BatchAttentionBackward(float * d_inputs,float const * d_outputs,float const * inputs,int B,int T ,int C3,int NH){

}

}