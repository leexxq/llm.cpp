#include <cute/tensor.hpp>
#include "config.cuh"
#include "online_softmax.cuh"
#include "log.cuh"
#include "utils.cuh"
using namespace cute;

// flash attention v2,see also https://arxiv.org/pdf/2307.08691
// visit https://blog.echen.io/p/flashattention-2-in-cute-from-scratch
template <class TQ, class LayoutQ, class TiledCopyQ,
		class TK, class LayoutK, class TiledCopyK,
		class TV, class LayoutV, class TiledCopyV,
		class TO, class LayoutO, class TiledCopyO,
		class TL, class LayoutL, 
		class TiledMMAS, class TiledMMAO,
		int Br = 64, int Bc = 64,int Hc = 32,AttentionType Attention= AttentionType::Default>
__global__ void AttentionForwardKernel(TQ const *Q, LayoutQ L_Q, TiledCopyQ copy_Q,
		TK const *K, LayoutK L_K, TiledCopyK copy_K,
		TV const *V, LayoutV L_V, TiledCopyV copy_V,
		TO *O, LayoutO L_O, TiledCopyO copy_O,
		TL *L, LayoutL L_L,
		TiledMMAS mmaS, TiledMMAO mmaO) {

	__shared__ TQ shared_memQ[Br * Hc];
	__shared__ TK shared_memK[Bc * Hc];
	__shared__ cute::half_t shared_memV[Bc * Hc];


	Tensor block_Q = make_tensor(Q, L_Q)(blockIdx.x, blockIdx.y, _, _); //(T,Hc)
	Tensor block_K = make_tensor(K, L_K)(blockIdx.x, blockIdx.y, _, _); //(T,Hc)
	Tensor block_V = make_tensor(V, L_V)(blockIdx.x, blockIdx.y, _, _); //(T,Hc)
	Tensor block_O = make_tensor(O, L_O)(blockIdx.x, blockIdx.y, _, _); //(T,Hc)
	Tensor block_L = make_tensor(L, L_L)(blockIdx.x, blockIdx.y,_); //(T)

	Tensor identity_coord_tensor = make_identity_tensor(make_shape(size<0>(block_Q),size<0>(block_K))); //(T,T)
	
	

	auto Tr =  size<0>(block_Q) / Br;
	auto Tc =  size<0>(block_K) / Bc;
	// auto Tr = 1;
	// auto Tc = 1;

	fwd_thread_print("Tr: %d, Tc : %d \n",Tr,Tc);
	for (int tr = 0; tr < Tr; ++tr) {
		Tensor gQ = local_tile(block_Q, make_tile(Int<Br>{}, Int<Hc>{}), make_coord(tr, 0));
		Tensor gO = local_tile(block_O, make_tile(Int<Br>{}, Int<Hc>{}), make_coord(tr, 0));


		ThrMMA thr_mmaS = mmaS.get_slice(threadIdx.x);
		//alloc fragment for S
		Tensor tSrS = partition_fragment_C(mmaS, make_shape(Int<Br>{}, Int<Bc>{}));

		// retile the register file layout (due per thread ownning multi row datas, we each calculate them.)
		Tensor row_tSrS = convert_C_to_row(tSrS); //(nums_pre_row,rows)

		OnlineSoftmax<Hc, size<1>(row_tSrS)> online_softmax;

		ThrMMA thr_mmaO = mmaO.get_slice(threadIdx.x);
		//alloc fragment for O
		Tensor tOrO = partition_fragment_C(mmaO, make_shape(Int<Br>{}, Int<Hc>{}));
		clear(tOrO);

		// retile the register file layout (due per thread ownning multi row datas, we each calculate them.)
		Tensor row_tOrO = convert_C_to_row(tOrO); //(nums_pre_row,rows)


		
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


		ThrCopy thr_copy_K = copy_K.get_slice(threadIdx.x);
		Tensor tKsK = thr_copy_K.partition_D(sK); //(CPY,CPY_T,CPY_Hc)
		Tensor gK_first = local_tile(block_K, make_tile(Int<Bc>{}, Int<Hc>{}), make_coord(0, 0));

		//Kcopy 
		Tensor tKgK_first = thr_copy_K.partition_S(gK_first); //(CPY,CPY_Bc,CPY_Hc)

		//issue q copy
		copy(copy_Q, tQgQ, tQsQ);
		// issue k copy
		copy(copy_K, tKgK_first, tKsK);
		// async copy q and k
		cp_async_fence();

		for (int tc = 0; tc < Tc; ++tc) {


			Tensor cta_identity_coord_tensor = local_tile(identity_coord_tensor,make_tile(Int<Br>{},Int<Bc>{}),make_coord(tr,tc));
			Tensor identity_coord_S_frag = thr_mmaS.partition_C(cta_identity_coord_tensor);
			Tensor row_identity_coord_S_frag = convert_C_to_row(identity_coord_S_frag);

			fwd_thread_print_tensor_verbose(identity_coord_S_frag);
			fwd_thread_print_tensor_verbose(row_identity_coord_S_frag);



			Tensor gV = local_tile(block_V, make_tile(Int<Bc>{}, Int<Hc>{}), make_coord(tc, 0));



			// wait previous k copy 
			cp_async_wait<0>();
			//!! must all thread can see smem latest 
			__syncthreads();


			// now gmem -> smem copy have been completed for Q and V
			// smem -> register follow ... 
			

			Tensor tSrQ = thr_mmaS.partition_fragment_A(sQ); // (MMA, MMA_Br,MMA_Hc)
			Tensor tSrK = thr_mmaS.partition_fragment_B(sK); // (MMA, MMA_Bc,MMA_Hc)

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

			fwd_thread_print_tensor("tSrQ:", tSrQ);
			fwd_thread_print_tensor("tSrK:", tSrK);


			// S need to be clear
			clear(tSrS);
			// S = QK^T
			CUTE_UNROLL
			for (int k = 0; k < size<2>(tSrQ); ++k) {
				gemm(mmaS, tSrQ(_, _, k), tSrK(_, _, k), tSrS);
			}

			fwd_thread_print_tensor("tSrS(gemm(Q,KT)):",tSrS);
			fwd_thread_print_tensor("row_tSrS(gemm(Q,KT)):",row_tSrS);


			// next K prefetch
			if(tc < Tc - 1){
				if constexpr (Attention == AttentionType::Causal) {
					// one block's bottom left not at mask
					if(tr * Br + Br - 1 < (tc + 1) * Bc ){
						fwd_thread_print("next block's causal condition not met, stop next k perfetch\n");
					}else{
						Tensor gK_next = local_tile(block_K, make_tile(Int<Bc>{}, Int<Hc>{}), make_coord(tc+1, 0));
						Tensor tKgK_next = thr_copy_K.partition_S(gK_next); //(CPY,CPY_Bc,CPY_Hc)
						// issue next K copy
						copy(copy_K,tKgK_next,tKsK);
						cp_async_fence();
					}
				}else{
					Tensor gK_next = local_tile(block_K, make_tile(Int<Bc>{}, Int<Hc>{}), make_coord(tc+1, 0));
					Tensor tKgK_next = thr_copy_K.partition_S(gK_next); //(CPY,CPY_Bc,CPY_Hc)
					// issue next K copy
					copy(copy_K,tKgK_next,tKsK);
					cp_async_fence();
				}
			}


			// online softmax
			if (tc == 0) {
				online_softmax.template scaled_softmax<true,Attention>(row_tSrS,row_tOrO,row_identity_coord_S_frag);
			} else {
				online_softmax.template scaled_softmax<false,Attention>(row_tSrS,row_tOrO,row_identity_coord_S_frag);
			}

			fwd_thread_print_tensor("tsum_frag:",online_softmax.tsums_frag);
			fwd_thread_print_tensor("tmax_frag:",online_softmax.tmaxs_frag);

			fwd_thread_print_tensor("tSrS(after scaled_softmax , S = e^(S - m) ) : ",tSrS);
			fwd_thread_print_tensor("row_tSrS (after scaled_softmax , S = e^(S - m) ) : ",row_tSrS);


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

				fwd_thread_print_tensor("gV:", gV);
				fwd_thread_print_tensor("tVgV:", tVgV);
				fwd_thread_print_tensor("tVrV:", tVrV);


				//convert to half type
				Tensor tVrV_fp16 = convert_type<cute::half_t>(tVrV);

				fwd_thread_print_tensor("tVrV fp16:",tVrV_fp16);

				Tensor tVsV = thr_copy_V.partition_D(sV);
				copy(copy_V, tXrV, tVsV);

				// !!! sV must copy finish
				__syncthreads();
			}

			fwd_thread_print_tensor("sV:",sV);

			Layout sVT_layout = make_layout(make_shape(Int<Hc>{}, Int<Bc>{}), LayoutLeft{});
			Tensor sVT = make_tensor(make_smem_ptr(shared_memV), sVT_layout);

			Copy_Atom<SM75_U16x8_LDSM_T, cute::half_t> s2r_atom_V;

			TiledCopy s2r_copy_VT = make_tiled_copy_B(s2r_atom_V, mmaO);
			ThrCopy s2r_thr_copy_VT = s2r_copy_VT.get_slice(threadIdx.x);


			Tensor tVrVT = thr_mmaO.partition_fragment_B(sVT); // (MMA, MMA_Bc,MMA_Hc)

			Tensor tXsVT = s2r_thr_copy_VT.partition_S(sVT);
			Tensor tXrVT = s2r_thr_copy_VT.retile_D(tVrVT);

			copy(s2r_atom_V, tXsVT, tXrVT);

			fwd_thread_print_tensor("sVT:",sVT);
			fwd_thread_print_tensor("tVrVT:",tVrVT);

			Tensor tSrS_fp16 = convert_type<cute::half_t>(tSrS);

			fwd_thread_print_tensor("tSrS:",tSrS);
			fwd_thread_print_tensor("tSrS_fp16",tSrS);

			// __syncthreads();
			// fwd_threadid_print_tensor(32,"tSrS:",tSrS);
			// fwd_threadid_print_tensor(32,"tSrS_fp16",tSrS);
			// __syncthreads();

			// convert s frag to a frag for o
			auto retile_shape = make_shape(make_shape(size<0, 0>(tSrS_fp16), size<0, 1>(tSrS_fp16), _2{}), size<1>(tSrS_fp16), size<2>(tSrS_fp16) / _2{});
			auto retile_stride = make_stride(make_stride(stride<0, 0>(tSrS_fp16), stride<0, 1>(tSrS_fp16), stride<2>(tSrS_fp16)), stride<1>(tSrS_fp16), stride<2>(tSrS_fp16) * _2{});


			Tensor tSrS_fp16_2_A_frag = make_tensor(tSrS_fp16.data(), make_layout(retile_shape, retile_stride));

			fwd_thread_print_tensor("tSrS_fp16_2_A_frag:",tSrS_fp16_2_A_frag);

			// __syncthreads();
			// fwd_threadid_print_tensor(32,"tSrS_fp16_2_A_frag:",tSrS_fp16_2_A_frag);
			// __syncthreads();


			// we will calculate SV
			// due O = SV
			CUTE_UNROLL
			for (int k = 0; k < size<2>(tVrVT); ++k) {
				gemm(mmaO, tSrS_fp16_2_A_frag(_, _, k), tVrVT(_, _, k), tOrO);
			}


			fwd_thread_print_tensor("tOrO (e^(S-m) * V):",tOrO);

			// __syncthreads();
			// fwd_threadid_print_tensor(32,"tOrO (e^(S-m) * V):",tOrO);
			// __syncthreads();
			if constexpr (Attention == AttentionType::Causal) {
				// one block's bottom left not at mask
				if(tr * Br + Br - 1 < (tc + 1) * Bc ){
					fwd_thread_print("next block's causal condition not met, break inner loop\n");
					break;
				}
			}

		}
		// scale o ...
		// normalize_scale_o(row_tOrO, tsum_frag);
		online_softmax.normalize_scale_o(row_tOrO);

		fwd_thread_print_tensor("tOrO(after normalize scale ...):",tOrO);

		Tensor O_identity_coord_tensor = local_tile(identity_coord_tensor,make_tile(Int<Br>{},Int<Hc>{}),make_coord(tr,0));
		Tensor identity_coord_O_frag = thr_mmaO.partition_C(O_identity_coord_tensor);
		Tensor row_identity_coord_O_frag = convert_C_to_row(identity_coord_O_frag);

		fwd_thread_print_tensor_verbose(identity_coord_O_frag);

		fwd_thread_print_tensor_verbose(row_identity_coord_O_frag);

		// write L
		online_softmax.write_logsumexp(block_L,row_identity_coord_O_frag);

		__syncthreads();
		fwd_thread_print_tensor_verbose(block_L);

		

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

		fwd_thread_print_tensor("tOrO:",tOrO);
		fwd_thread_print_tensor("sO:",sO);

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


		fwd_thread_print_tensor("gO(after copy):",gO);
		fwd_thread_print_tensor("copy_tOrO_frag:",copy_tOrO_frag);
		fwd_thread_print_tensor("copy_tOgO:",copy_tOgO);
	}
}