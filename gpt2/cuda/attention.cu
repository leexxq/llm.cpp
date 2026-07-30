#include "attention.cuh"
#include "cute/arch/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/config.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/stride.hpp"
#include "cute/tensor_impl.hpp"
#include "cute/util/print.hpp"
#include "global.cuh"
#include "cute/tensor.hpp"
#include "cute/util/debug.hpp"
#include "cute/util/print_tensor.hpp"
#include "cutlass/numeric_conversion.h"
#include "cutlass/util/device_memory.h"
#include "math_constants.h"
#include <cassert>
#include <cstdio>
namespace gpt2cuda {
namespace kernel {
using namespace cute;

// 1 is print 
#if 0
	#define debug_thread 0

	#define threadid_print(id,...) do{\
			if(thread(id)){\
				printf("[thread:%d]",id);\
				printf(__VA_ARGS__);\
			}\
		}while(false)

	#define threadid_print_tensor(id,desc,tensor) do{\
			threadid_print(id,desc);\
			if(thread(id)){\
				print_tensor(tensor);\
			}\
		}while(false)

	#define thread_print(...) threadid_print(debug_thread,__VA_ARGS__)

	#define thread_print_tensor(desc,tensor) threadid_print_tensor(debug_thread,desc,tensor)
	#define thread_print_tensor_verbose(tensor) thread_print_tensor(#tensor" :",tensor)


#else
	#define thread_print(...)
	#define thread_print_tensor(x,y)
	#define thread_print_tensor_verbose(tensor)

#endif

enum class AttentionType{
	Default,
	Causal,
};

template  <int Hc , int N> 
struct OnlineSoftmax{


	constexpr static float klog_2e = 1.4426950409f;
	constexpr static float kscale = Hc==32 ? 0.176777:0.125f; // 1 / sqrt(32);
	constexpr static float klog2_scale = klog_2e * kscale;

	// alloc register for sum op and max op;
	// Tensor tsum_frag = make_tensor<float>(Int<size<1>(row_tSrS)>{});
	// Tensor tmax_frag = make_tensor<float>(Int<size<1>(row_tSrS)>{});

	using Tensor_reg = decltype(make_tensor<float>(Int<N>{}));

	Tensor_reg tsums_frag;
	Tensor_reg tmaxs_frag;

	template <bool Is_first, AttentionType Attention, class S_tensor, class O_tensor, class M_tensor >
	__forceinline__ __device__ void scaled_softmax(S_tensor& row_tSrS, O_tensor& row_tOrO,M_tensor& row_identity_coord){

		CUTE_STATIC_ASSERT_V(size<1>(row_tOrO) == size<1>(row_tSrS));


		// int i = get<0>(row_mask(2,1)) , j = get<1>(row_mask(2,1));
		// thread_print("row_mask(2,1) : (i : %d , j : %d) \n" , i, j);



		CUTE_UNROLL
		for (int r = 0; r < size<1>(row_tSrS); ++r) {
			float tmax_frag_new = -CUDART_INF_F;

			CUTE_UNROLL
			for (int c = 0; c < size<0>(row_tSrS); ++c) {
				// per thread per max op
				if constexpr (Attention == AttentionType::Causal) {
					auto [i,j] = row_identity_coord(c,r);
					tmax_frag_new = j > i ? tmax_frag_new :  max(tmax_frag_new, row_tSrS(c, r));
				}else {
					tmax_frag_new = max(tmax_frag_new, row_tSrS(c, r));
				}
			}

			// quad reduce max
			CUTE_UNROLL
			for (int off = 1; off < 4; off <<= 1) {
				tmax_frag_new = max(__shfl_xor_sync(0xFFFFFFFFU, tmax_frag_new, off), tmax_frag_new);
			}

			if constexpr (!Is_first) {
				tmax_frag_new = max(tmaxs_frag(r), tmax_frag_new);
			}

			// scale_softmax : softmax(QK^T);
			CUTE_UNROLL
			for (int c = 0; c < size<0>(row_tSrS); ++c) {
				if constexpr (Attention == AttentionType::Causal) {
					auto [i,j] = row_identity_coord(c,r);
					row_tSrS(c, r) = j > i ? 0.f : exp2f(klog2_scale * (row_tSrS(c, r) - tmax_frag_new));
				}else {
					row_tSrS(c, r) = exp2f(klog2_scale * (row_tSrS(c, r) - tmax_frag_new));
				}
			}


			float tsum_frag_new = 0.f;


			CUTE_UNROLL
			for (int c = 0; c < size<0>(row_tSrS); ++c) {
				// per thread sum
				tsum_frag_new += row_tSrS(c, r);
			}

			// quad reduce sum
			CUTE_UNROLL
			for (int off = 1; off < 4; off <<= 1) {
				tsum_frag_new += __shfl_xor_sync(0xFFFFFFFFU, tsum_frag_new, off);
			}

			// per thread sum
			if constexpr (!Is_first) {
				float correction = exp2f(klog2_scale * (tmaxs_frag(r) - tmax_frag_new));
				tsum_frag_new += correction * tsums_frag(r);
				// scale pre o
				CUTE_UNROLL
				for (int c = 0; c < size<0>(row_tOrO); ++c) {
					row_tOrO(c, r) = correction * row_tOrO(c, r);
				}
			}

			tmaxs_frag(r) = tmax_frag_new;
			tsums_frag(r) = tsum_frag_new;
		}
	}

	template <class O_tensor>
	__forceinline__ __device__ void normalize_scale_o(O_tensor& row_tOrO) {
		CUTE_UNROLL
		for (int r = 0; r < size<1>(row_tOrO); ++r) {
			float norm = 1.f/ tsums_frag(r);
			for (int c = 0; c < size<0>(row_tOrO); ++c) {
				row_tOrO(c, r) = row_tOrO(c, r) * norm;
			}
		}
	}


};


template <class To_type, class Engine, class Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
	using From_type = typename Engine::value_type;

	Tensor target = make_fragment_like<To_type>(tensor.layout());

	auto convert_op = cutlass::NumericConverter<To_type, From_type>{};

	CUTE_UNROLL
	for (int i = 0; i < size(tensor); ++i) {
		target(i) = convert_op(tensor(i));
	}
	return target;
}





// Here, "row" means that I place a row of data owned by a thread at the first index,
// while the second index corresponds to other rows.
template <class Engine,class Layout>
__forceinline__ __device__ auto convert_C_to_row(Tensor<Engine,Layout> & ctensor) {
	// retile the register file layout (due per thread ownning multi row datas, we each calculate them.)

	auto row_tiler = make_layout(
			make_shape(size<0, 0>(ctensor), Int<1>{}, size<2>(ctensor)),
			make_stride(Int<1>{}, Int<0>{}, size<0>(ctensor) * size<1>(ctensor)));
	return make_tensor(ctensor.data(),zipped_divide(ctensor.layout(), row_tiler)); //(nums_pre_row,rows)
}




// flash attention v2,see also https://arxiv.org/pdf/2307.08691
// visit https://blog.echen.io/p/flashattention-2-in-cute-from-scratch
template <class TQ, class LayoutQ, class TiledCopyQ,
		class TK, class LayoutK, class TiledCopyK,
		class TV, class LayoutV, class TiledCopyV,
		class TO, class LayoutO, class TiledCopyO,
		class TiledMmaS, class TiledMmaO,
		int Br = 64, int Bc = 64,int Hc = 32,AttentionType Attention= AttentionType::Default>
__global__ void AttentionForwardKernel(TQ const *Q, LayoutQ L_Q, TiledCopyQ copy_Q,
		TK const *K, LayoutK L_K, TiledCopyK copy_K,
		TV const *V, LayoutV L_V, TiledCopyV copy_V,
		TO *O, LayoutO L_O, TiledCopyO copy_O,
		TiledMmaS mmaS, TiledMmaO mmaO) {

	__shared__ float shared_memQ[Br * Hc];

	__shared__ float shared_memK[Bc * Hc];

	__shared__ cute::half_t shared_memV[Bc * Hc];


	auto block_Q = make_tensor(Q, L_Q)(blockIdx.x, blockIdx.y, _, _); //(T,Hc)
	auto block_K = make_tensor(K, L_K)(blockIdx.x, blockIdx.y, _, _); //(T,Hc)
	auto block_V = make_tensor(V, L_V)(blockIdx.x, blockIdx.y, _, _); //(T,Hc)
	auto block_O = make_tensor(O, L_O)(blockIdx.x, blockIdx.y, _, _); //(T,Hc)

	Tensor identity_coord_tensor = make_identity_tensor(make_shape(size<0>(block_Q),size<0>(block_K))); //(T,T)
	
	

	auto Tr =  size<0>(block_Q) / Br;
	auto Tc =  size<0>(block_K) / Bc;
	// auto Tr = 1;
	// auto Tc = 1;

	thread_print("Tr: %d, Tc : %d \n",Tr,Tc);
	for (int tr = 0; tr < Tr; ++tr) {
		Tensor gQ = local_tile(block_Q, make_tile(Int<Br>{}, Int<Hc>{}), make_coord(tr, 0));
		Tensor gO = local_tile(block_O, make_tile(Int<Br>{}, Int<Hc>{}), make_coord(tr, 0));

		//alloc fragment for S
		Tensor tSrS = partition_fragment_C(mmaS, make_shape(Int<Br>{}, Int<Bc>{}));

		// retile the register file layout (due per thread ownning multi row datas, we each calculate them.)
		Tensor row_tSrS = convert_C_to_row(tSrS); //(nums_pre_row,rows)


		ThrMMA thr_mmaS = mmaS.get_slice(threadIdx.x);


		OnlineSoftmax<Hc, size<1>(row_tSrS)> online_softmax;

		//alloc fragment for O
		Tensor tOrO = partition_fragment_C(mmaO, make_shape(Int<Br>{}, Int<Hc>{}));
		clear(tOrO);

		// retile the register file layout (due per thread ownning multi row datas, we each calculate them.)
		Tensor row_tOrO = convert_C_to_row(tOrO); //(nums_pre_row,rows)


		for (int tc = 0; tc < Tc; ++tc) {
			Tensor cta_identity_coord_tensor = local_tile(identity_coord_tensor,make_tile(Int<Br>{},Int<Bc>{}),make_coord(tr,tc));

			Tensor identity_coord_S_frag = thr_mmaS.partition_C(cta_identity_coord_tensor);
			Tensor row_identity_coord_S_frag = convert_C_to_row(identity_coord_S_frag);

			thread_print_tensor_verbose(identity_coord_S_frag);
			thread_print_tensor_verbose(row_identity_coord_S_frag);

			if constexpr (Attention == AttentionType::Causal) {
				// one block's bottom left not at mask
				if(tr * Br + Br - 1 < tc * Bc ){
					break;
				}
			}

			thread_print("tr:%d,tc:%d\n",tr,tc);

			Tensor gK = local_tile(block_K, make_tile(Int<Bc>{}, Int<Hc>{}), make_coord(tc, 0));
			Tensor gV = local_tile(block_V, make_tile(Int<Bc>{}, Int<Hc>{}), make_coord(tc, 0));

			// print_tensor(thr_tile_S);

			//Qmem
			Layout sQ_layout = make_layout(make_shape(Int<Br>{}, Int<Hc>{}), LayoutRight{});
			Tensor sQ = make_tensor(make_smem_ptr(shared_memQ), sQ_layout); //(T,Hc,k)

			//Kmem
			Layout sK_layout = make_layout(make_shape(Int<Bc>{}, Int<Hc>{}), LayoutRight{});
			Tensor sK = make_tensor(make_smem_ptr(shared_memK), sK_layout); //(T,Hc,k)

			//Qcopy
			ThrCopy thr_copy_Q = copy_Q.get_slice(threadIdx.x);
			Tensor tQgQ = thr_copy_Q.partition_S(gQ); //(CPY,CPY_Br,CPY_Hc)
			Tensor tQsQ = thr_copy_Q.partition_D(sQ); //(CPY,CPY_T,CPY_Hc)

			//Kcopy
			ThrCopy thr_copy_K = copy_K.get_slice(threadIdx.x);
			Tensor tKgK = thr_copy_K.partition_S(gK); //(CPY,CPY_Bc,CPY_Hc)
			Tensor tKsK = thr_copy_K.partition_D(sK); //(CPY,CPY_T,CPY_Hc)

			// async copy q and k
			if (tc == 0) {
				copy(copy_Q, tQgQ, tQsQ);
			}

			copy(copy_K, tKgK, tKsK);
			cp_async_fence();

			//wait q and k
			cp_async_wait<0>();
			//!! must all thread can see smem latest 
			__syncthreads();

			ThrMMA thr_mmaS = mmaS.get_slice(threadIdx.x);

			Tensor tSrQ = thr_mmaS.partition_fragment_A(sQ); // (MMA, MMA_Br,MMA_Hc)
			Tensor tSrK = thr_mmaS.partition_fragment_B(sK); // (MMA, MMA_Br,MMA_Hc)

			// Copy_Atom<UniversalCopy<float>, float> s2r_atom_Q;
			// Copy_Atom<UniversalCopy<float>, float> s2r_atom_K;

			Copy_Atom<SM75_U32x4_LDSM_N, float> s2r_atom_Q;
			Copy_Atom<SM75_U32x4_LDSM_N, float> s2r_atom_K;
			TiledCopy s2r_copy_Q = make_tiled_copy_A(s2r_atom_Q, mmaS);
			ThrCopy s2r_thr_copy_Q = s2r_copy_Q.get_slice(threadIdx.x);

			Tensor tXsQ = s2r_thr_copy_Q.partition_S(sQ);
			Tensor tXrQ = s2r_thr_copy_Q.retile_D(tSrQ);

			TiledCopy s2r_copy_K = make_tiled_copy_B(s2r_atom_K, mmaS);
			ThrCopy s2r_thr_copy_K = s2r_copy_K.get_slice(threadIdx.x);

			Tensor tXsK = s2r_thr_copy_K.partition_S(sK); //(CPY,MMA_T,MMA_Hc)
			Tensor tXrK = s2r_thr_copy_K.retile_D(tSrK);

			copy(s2r_atom_Q, tXsQ, tXrQ);
			copy(s2r_atom_K, tXsK, tXrK);

			thread_print_tensor("tSrQ:", tSrQ);
			thread_print_tensor("tSrK:", tSrK);
			


			// S need to be clear
			clear(tSrS);
			// S = QK^T
			for (int k = 0; k < size<2>(tSrQ); ++k) {
				gemm(mmaS, tSrQ(_, _, k), tSrK(_, _, k), tSrS);
			}

			thread_print_tensor("tSrS(gemm(Q,KT)):",tSrS);
			thread_print_tensor("row_tSrS(gemm(Q,KT)):",row_tSrS);





			// online softmax
			if (tc == 0) {
				online_softmax.template scaled_softmax<true,Attention>(row_tSrS,row_tOrO,row_identity_coord_S_frag);
			} else {
				online_softmax.template scaled_softmax<false,Attention>(row_tSrS,row_tOrO,row_identity_coord_S_frag);
			}

			thread_print_tensor("tsum_frag:",online_softmax.tsums_frag);
			thread_print_tensor("tmax_frag:",online_softmax.tmaxs_frag);

			thread_print_tensor("tSrS(after scaled_softmax , S = e^(S - m) ) : ",tSrS);
			thread_print_tensor("row_tSrS (after scaled_softmax , S = e^(S - m) ) : ",row_tSrS);


			//now we get per row from formula softmax(QK^T)

			// V smem (swizzle)
			Layout sV_layout = make_layout(make_shape(Int<Bc>{}, Int<Hc>{}), LayoutRight{});
			// Tensor sV_swz = make_tensor(make_smem_ptr(shared_memV),composition(Swizzle<5,0,5>{},sV_layout));
			Tensor sV = make_tensor(make_smem_ptr(shared_memV), sV_layout);
			ThrCopy thr_copy_V = copy_V.get_slice(threadIdx.x);

			Tensor tVgV = thr_copy_V.partition_S(gV);

			// copy global to register and convert fp32 to fp16.
			// finally, copy to smem
			{
				Tensor tVrV = make_fragment_like<float>(tVgV);
				Tensor tXrV = thr_copy_V.retile_D(tVrV);
				copy(copy_V, tVgV, tXrV);

				thread_print_tensor("gV:", gV);
				thread_print_tensor("tVgV:", tVgV);
				thread_print_tensor("tVrV:", tVrV);


				//convert to half type
				Tensor tVrV_fp16 = convert_type<cute::half_t>(tVrV);

				thread_print_tensor("tVrV fp16:",tVrV_fp16);

				Tensor tVsV = thr_copy_V.partition_D(sV);
				copy(copy_V, tXrV, tVsV);

				// !!! sV must copy finish
				__syncthreads();
			}

			thread_print_tensor("sV:",sV);

			Layout sVT_layout = make_layout(make_shape(Int<Hc>{}, Int<Bc>{}), LayoutLeft{});
			Tensor sVT = make_tensor(make_smem_ptr(shared_memV), sVT_layout);

			Copy_Atom<SM75_U16x8_LDSM_T, cute::half_t> s2r_atom_V;

			TiledCopy s2r_copy_VT = make_tiled_copy_B(s2r_atom_V, mmaO);
			ThrCopy s2r_thr_copy_VT = s2r_copy_VT.get_slice(threadIdx.x);

			ThrMMA thr_mmaO = mmaO.get_slice(threadIdx.x);

			Tensor tVrVT = thr_mmaO.partition_fragment_B(sVT); // (MMA, MMA_Bc,MMA_Hc)

			Tensor tXsVT = s2r_thr_copy_VT.partition_S(sVT);
			Tensor tXrVT = s2r_thr_copy_VT.retile_D(tVrVT);

			copy(s2r_atom_V, tXsVT, tXrVT);

			thread_print_tensor("sVT:",sVT);
			thread_print_tensor("tVrVT:",tVrVT);

			Tensor tSrS_fp16 = convert_type<cute::half_t>(tSrS);

			thread_print_tensor("tSrS:",tSrS);
			thread_print_tensor("tSrS_fp16",tSrS);

			// __syncthreads();
			// threadid_print_tensor(32,"tSrS:",tSrS);
			// threadid_print_tensor(32,"tSrS_fp16",tSrS);
			// __syncthreads();

			// convert s frag to a frag for o
			auto retile_shape = make_shape(make_shape(size<0, 0>(tSrS_fp16), size<0, 1>(tSrS_fp16), _2{}), size<1>(tSrS_fp16), size<2>(tSrS_fp16) / _2{});
			auto retile_stride = make_stride(make_stride(stride<0, 0>(tSrS_fp16), stride<0, 1>(tSrS_fp16), stride<2>(tSrS_fp16)), stride<1>(tSrS_fp16), stride<2>(tSrS_fp16) * _2{});


			Tensor tSrS_fp16_2_A_frag = make_tensor(tSrS_fp16.data(), make_layout(retile_shape, retile_stride));

			thread_print_tensor("tSrS_fp16_2_A_frag:",tSrS_fp16_2_A_frag);

			// __syncthreads();
			// threadid_print_tensor(32,"tSrS_fp16_2_A_frag:",tSrS_fp16_2_A_frag);
			// __syncthreads();





			// we will calculate SV
			// due O = SV
			for (int k = 0; k < size<2>(tVrVT); ++k) {
				gemm(mmaO, tSrS_fp16_2_A_frag(_, _, k), tVrVT(_, _, k), tOrO);
			}


			thread_print_tensor("tOrO (e^(S-m) * V):",tOrO);

			// __syncthreads();
			// threadid_print_tensor(32,"tOrO (e^(S-m) * V):",tOrO);
			// __syncthreads();

		}
		// scale o ...
		
		// normalize_scale_o(row_tOrO, tsum_frag);
		online_softmax.template normalize_scale_o<decltype(row_tOrO)>(row_tOrO);

		thread_print_tensor("tOrO(after normalize scale ...):",tOrO);

		// copy O to Q for coalease write
		Layout sO_layout = make_layout(make_shape(Int<Br>{}, Int<Hc>{}), LayoutRight{});
		Tensor sO = make_tensor(make_smem_ptr(shared_memQ), sO_layout);

		using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopy, float>;

		TiledCopy r2s_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, mmaO);

		ThrCopy smem_thr_copy_O = r2s_tiled_copy_O.get_slice(threadIdx.x);

		Tensor copy_trO = smem_thr_copy_O.retile_S(tOrO);
		Tensor copy_tsO = smem_thr_copy_O.partition_D(sO);

		// register -> smem
		copy(r2s_tiled_copy_O, copy_trO, copy_tsO);

		// sO must copy finish!
		__syncthreads();

		thread_print_tensor("tOrO:",tOrO);
		thread_print_tensor("sO:",sO);

		ThrCopy s2g_thr_tiled_copy = copy_O.get_slice(threadIdx.x);
		Tensor copy_tOsO = s2g_thr_tiled_copy.partition_S(sO);
		Tensor copy_tOgO = s2g_thr_tiled_copy.partition_D(gO);

		Tensor copy_tOrO_frag = make_fragment_like(copy_tOgO);
		Tensor copy_tOrO_frag_retile = s2g_thr_tiled_copy.retile_D(copy_tOrO_frag);

		// smem -> registers
		copy(copy_O, copy_tOsO, copy_tOrO_frag_retile);

		// if (thread0()) {
		// 	print("copy_tOsO:");
		// 	print_tensor(copy_tOsO);
		// 	print("copy_tOrO_frag:");
        //     print_tensor(copy_tOrO_frag);
		// }

        // if(thread0()){
        //     print("gO:");
        //     print_tensor(gO);
        // }

		// registers -> gmem
		copy(copy_O,copy_tOrO_frag_retile,copy_tOgO);


		thread_print_tensor("gO(after copy):",gO);
		thread_print_tensor("copy_tOrO_frag:",copy_tOrO_frag);
		thread_print_tensor("copy_tOgO:",copy_tOgO);
	}
}

template<AttentionType Attention ,int Hc, int Br = 64 , int Bc = 64 >
void AttentionForwardCUDA(float *outputs, float const *inputs, int B, int T, int C3, int NH) {

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
			decltype(mmaS), decltype(mmaO),
			Br,Bc,Hc,Attention>;


	
	dim3 dimGrid(B,NH);
	kernel_fptr<<<dimGrid, 128>>>(inputs, L_Q, copyQ, inputs + C, L_K, copyK, inputs + 2 * C, L_V, copyV, outputs, L_O, copyO, mmaS, mmaO);
    CUDA_CHECK_LAST();
}

} //namespace kernel

using DAlloc = cutlass::device_memory::allocation<float>;

void BatchAttentionForward(float *outputs, float const *inputs, kernel::AttentionType Attention , int B, int T, int C3, int NH) {
	using namespace cute;
	assert(C3 % 3 == 0);
	auto C = C3 / 3;

	DAlloc outputs_d(B * T * C);
	DAlloc inputs_d(B * T * C3);

	outputs_d.copy_from_host(outputs);
	inputs_d.copy_from_host(inputs);

	if(C/NH == 32){
		if(Attention == kernel::AttentionType::Default){
			kernel::AttentionForwardCUDA<kernel::AttentionType::Default,32>(outputs_d.get(), inputs_d.get(), B, T, C3, NH);
		}else if(Attention == kernel::AttentionType::Causal){
			kernel::AttentionForwardCUDA<kernel::AttentionType::Causal,32>(outputs_d.get(), inputs_d.get(), B, T, C3, NH);
		}else{
			std::cerr << "fatal: " << static_cast<int>(Attention) << " not exists!"<< std::endl;
			exit(1);
		}
	}else if(C/NH == 64){
		if(Attention == kernel::AttentionType::Default){
			kernel::AttentionForwardCUDA<kernel::AttentionType::Default,64>(outputs_d.get(), inputs_d.get(), B, T, C3, NH);
		}else if(Attention == kernel::AttentionType::Causal){
			kernel::AttentionForwardCUDA<kernel::AttentionType::Causal,64>(outputs_d.get(), inputs_d.get(), B, T, C3, NH);
		}else{

		}
	}else
	{
		std::cerr << "not supported attention shape D = " << C/NH << std::endl;
		exit(1);
	}

	outputs_d.copy_to_host(outputs);
}

void BatchAttentionBackward(float *d_inputs, float const *d_outputs, float const *inputs, int B, int T, int C3, int NH) {
}

void BatchAttentionForward(float *outputs, float const *inputs, int B, int T, int C3, int NH) {
	BatchAttentionForward(outputs,inputs,kernel::AttentionType::Default , B,T,C3,NH);
}

void BatchCausalAttentionBackward(float *d_inputs, float const *d_outputs, float const *inputs, int B, int T, int C3, int NH) {
}


void BatchCausalAttentionForward(float *outputs, float const *inputs, int B, int T, int C3, int NH) {
	BatchAttentionForward(outputs,inputs,kernel::AttentionType::Causal , B,T,C3,NH);
}


} //namespace gpt2cuda