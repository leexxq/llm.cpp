#include <cute/tensor.hpp>
#include "cutlass/bfloat16.h"
#include "online_softmax.cuh"
#include "log.cuh"
#include "utils.cuh"
#include "config.cuh"
// compute D = rowsum(O o dO)
template<
			class Config,
			class LayoutO,class LayoutdO,class LayoutD
		>
__device__ __forceinline__ void ComputeD(
					typename Config::TO const * O, LayoutO L_O, typename Config::CopyO C_O,
					typename Config::TdO const * dO, LayoutdO L_dO,
					typename Config::TD * D, LayoutD L_D
){
	constexpr int Br = Config::kBr;
	constexpr int Hc = Config::kHc;
	//per block compute 
	Tensor block_O = make_tensor(O,L_O)(blockIdx.x,blockIdx.y,_,_);//(T,Hc)
	Tensor block_dO = make_tensor(dO,L_dO)(blockIdx.x,blockIdx.y,_,_);//(T,Hc)
	Tensor block_D = make_tensor(D,L_D)(blockIdx.x,blockIdx.y,_);//(T)

	int Tr = size<0>(block_D) / Br;


	for(int tr = 0 ; tr < Tr ; ++tr){
		Tensor gO = local_tile(block_O,make_tile(Int<Br>{},Int<Hc>{}),make_coord(tr,0));
		Tensor gdO = local_tile(block_dO,make_tile(Int<Br>{},Int<Hc>{}),make_coord(tr,0));
		Tensor gD = local_tile(block_D,make_tile(Int<Br>{}),make_coord(tr));

		bwd_thread_print_tensor_verbose(gO);
		bwd_thread_print_tensor_verbose(gdO);

		ThrCopy thr_copy_O = C_O.get_slice(threadIdx.x);
		
		Tensor tOgO = thr_copy_O.partition_S(gO);//((Copy_X,Copy_Y),Copy_T,Copy_Hc)
		Tensor tOrO = thr_copy_O.retile_D(make_fragment_like<float>(tOgO));//((Copy_X,Copy_Y),Copy_T,Copy_Hc)

		Tensor local_o_identity_coord = make_identity_tensor(make_shape(Int<Br>{},Int<Hc>{}));
		Tensor o_identity_coord_frag = thr_copy_O.partition_S(local_o_identity_coord);

		ThrCopy thr_copy_dO = C_O.get_slice(threadIdx.x);
		
		Tensor tdOgdO = thr_copy_dO.partition_S(gdO);//((Copy_X,Copy_Y),Copy_T,Copy_Hc)
		Tensor tdOrdO = thr_copy_dO.retile_D(make_fragment_like<float>(tdOgdO));//((Copy_X,Copy_Y),Copy_T,Copy_Hc)

		copy(C_O,tOgO,tOrO);
		copy(C_O,tdOgdO,tdOrdO);

		bwd_thread_print_tensor_verbose(o_identity_coord_frag);
		bwd_thread_print_tensor_verbose(tOgO);
		bwd_thread_print_tensor_verbose(tOrO);
		bwd_thread_print_tensor_verbose(tdOgdO);
		bwd_thread_print_tensor_verbose(tdOrdO);

		Tensor row_tOrO = convert_to_row_for_D(tOrO);
		bwd_thread_print_tensor_verbose(row_tOrO);
		Tensor row_tdOrdO = convert_to_row_for_D(tdOrdO);
		bwd_thread_print_tensor_verbose(row_tdOrdO);
		Tensor row_o_identity_coord_frag = convert_to_row_for_D(o_identity_coord_frag);
		bwd_thread_print_tensor_verbose(row_o_identity_coord_frag);

		
		Tensor tDrD = make_tensor<typename Config::TD>(size<1>(row_tOrO));
		
		clear(tDrD);
		constexpr int threads_per_row = Config::kthreads_per_row;

		bwd_thread_print("threads pre row : %d \n",threads_per_row);
		
		for(int r = 0 ; r < size<1>(row_tOrO); ++r){
			for(int c= 0 ; c < size<0>(row_tOrO);++c){
				tDrD(r) += row_tOrO(c,r) * row_tdOrdO(c,r);
			}
			
			// 8 way reduce
			for(int off = 1 ; off < threads_per_row; off<<=1 ){
				tDrD(r) += __shfl_xor_sync(0xFFFFFFFFU,tDrD(r),off);
			}

			if(threadIdx.x % threads_per_row == 0){
				auto[i,j] = row_o_identity_coord_frag(0,r);
				gD(i) = tDrD(r); 
			}
		}
		bwd_thread_print_tensor_verbose(tDrD);
	}
	bwd_thread_print_tensor_verbose(block_D);

}


template<class Config,class Layouts>
__global__ void AttentionBackwardKernel(typename Config::TQ const * Q, typename Layouts::LayoutQ L_Q,
										typename Config::TK const * K, typename Layouts::LayoutK L_K, 
										typename Config::TV const * V, typename Layouts::LayoutV L_V, 
										typename Config::TO const * O, typename Layouts::LayoutO L_O, 
										typename Config::TdO const * dO, typename Layouts::LayoutdO L_dO, 
										typename Config::TL const * L, typename Layouts::LayoutL L_L,
										typename Config::TD * D, typename Layouts::LayoutD L_D, 			
										typename Config::TdQ  * dQ, typename Layouts::LayoutdQ L_dQ,
										typename Config::TdK  * dK, typename Layouts::LayoutdK L_dK,
										typename Config::TdV  * dV, typename Layouts::LayoutdV L_dV	
										){

											

    //trait config arguments
	constexpr int Br = Config::kBr;
	constexpr int Bc = Config::kBc;
	constexpr int Hc = Config::kHc;

    // Mma 
	typename Config::MMA_S mmaS,mmadP;
	typename Config::MMA_dQ mmadQ;
	typename Config::MMA_dK mmadK,mmadV;

 	typename Config::CopyQ C_Q;
	typename Config::CopyK C_K; 
	typename Config::CopyV C_V;
	typename Config::CopyO C_O;
	typename Config::CopydO C_dO;
    typename Config::CopyL C_L;
	typename Config::CopyD C_D;
	typename Config::CopydQ C_dQ;
	typename Config::CopydK C_dK;
	typename Config::CopydV C_dV;

	// Shared memory buffers
	extern __shared__ char shared_memory[];
	using SharedStorage = typename Config::SharedStorage;
	SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);

	auto smemQ = smem.Q.begin();
	auto smemK = smem.K.begin();
	auto smemV = smem.V.begin();

	auto smemdK = smem.dK.begin();
	auto smemdV = smem.dV.begin();

	auto smemP = smem.P.begin();
	auto smemdO = smem.dO.begin();

	auto smemL = smem.L.begin();
	auto smemD = smem.D.begin();

	//compute rowsum(O o dO) 
	ComputeD<Config>(O,L_O,C_O,dO,L_dO,D,L_D);
	__syncthreads();

	Tensor block_Q = make_tensor(make_gmem_ptr(Q),L_Q)(blockIdx.x,blockIdx.y,_,_);//(T,Hc)
	Tensor block_K = make_tensor(make_gmem_ptr(K),L_K)(blockIdx.x,blockIdx.y,_,_);//(T,Hc)
	Tensor block_V = make_tensor(make_gmem_ptr(V),L_V)(blockIdx.x,blockIdx.y,_,_);//(T,Hc)
	Tensor block_O = make_tensor(make_gmem_ptr(O),L_O)(blockIdx.x,blockIdx.y,_,_);//(T,Hc)
	Tensor block_dO = make_tensor(make_gmem_ptr(dO),L_dO)(blockIdx.x,blockIdx.y,_,_);//(T,Hc)

	Tensor block_dQ = make_tensor(make_gmem_ptr(dQ),L_dQ)(blockIdx.x,blockIdx.y,_,_);//(T,Hc)
	Tensor block_dK = make_tensor(make_gmem_ptr(dK),L_dK)(blockIdx.x,blockIdx.y,_,_);//(T,Hc)
	Tensor block_dV = make_tensor(make_gmem_ptr(dV),L_dV)(blockIdx.x,blockIdx.y,_,_);//(T,Hc)


	Tensor block_L = make_tensor(make_gmem_ptr(L),L_L)(blockIdx.x,blockIdx.y,_);//(T)
	Tensor block_D = make_tensor(make_gmem_ptr(D),L_D)(blockIdx.x,blockIdx.y,_);//(T)

	Tensor s_identity_coord_tensor  = make_identity_tensor(make_shape(size<0>(block_Q),size<0>(block_Q)));//(T,T)

	// CUTE_STATIC_ASSERT_V(size<0>(block_Q) == size<0>(block_K));
	// CUTE_STATIC_ASSERT_V(size<0>(block_Q) == size<0>(block_V));
	// CUTE_STATIC_ASSERT_V(size<0>(block_Q) == size<0>(block_O));
	// CUTE_STATIC_ASSERT_V(size<0>(block_Q) == size<0>(block_L));
	// CUTE_STATIC_ASSERT_V(size<0>(block_Q) == size<0>(block_D));

	int Tr = size<0>(block_Q) / Br;
	int Tc = size<0>(block_Q) / Bc;
	for(int tc = 0 ; tc < Tc ; ++tc){

		auto sKV_layout = make_layout(make_shape(Int<Bc>{},Int<Hc>{}),LayoutRight{});
		Tensor sK = make_tensor(make_smem_ptr(smemK),sKV_layout);
		Tensor sV = make_tensor(make_smem_ptr(smemV),sKV_layout);
		Tensor sdK = make_tensor(make_smem_ptr(smemdK),sKV_layout);
		Tensor sdV = make_tensor(make_smem_ptr(smemdV),sKV_layout);

		// load K, load V 
		
		Tensor gK = local_tile(block_K,make_tile(Int<Bc>{},Int<Hc>{}),make_coord(tc,0));
		Tensor gV = local_tile(block_V,make_tile(Int<Bc>{},Int<Hc>{}),make_coord(tc,0));


		//copy V,K to smem and convert to half

		vec_cp_g2s_to_half<cute::bfloat16_t>(gK,sK,C_K);
		vec_cp_g2s_to_half<cute::bfloat16_t>(gV,sV,C_V);



		ThrMMA thr_mmadK = mmadK.get_slice(threadIdx.x);
		Tensor tdKrdK = thr_mmadK.partition_fragment_C(sdK);

		ThrMMA thr_mmadV = mmadV.get_slice(threadIdx.x);
		Tensor tdVrdV = thr_mmadV.partition_fragment_C(sdV);

		clear(tdKrdK);
		clear(tdVrdV);

		for(int tr = 0; tr < Tr ; ++tr){
			
			ThrMMA thr_mmaS = mmaS.get_slice(threadIdx.x);
			ThrMMA thr_mmadP = mmadP.get_slice(threadIdx.x);

			if constexpr (Config::Attention == AttentionType::Causal) {
				if(tr * Br + Br - 1 < tc * Bc ){
					continue;
				}
			}

			//alloc fragment for S
			Tensor tSrS = partition_fragment_C(mmaS, make_shape(Int<Br>{}, Int<Bc>{}));
			//alloc fragment for dP
			Tensor tdPrdP = partition_fragment_C(mmadP, make_shape(Int<Br>{}, Int<Bc>{}));


			Tensor gQ = local_tile(block_Q,make_tile(Int<Br>{},Int<Hc>{}),make_coord(tr,0));

			auto sQ_layout = make_layout(make_shape(Int<Br>{},Int<Hc>{}),LayoutRight{});

			Tensor sQ = make_tensor(make_smem_ptr(smemQ),sQ_layout);
			//Qcopy
			vec_cp_g2s_to_half<cute::bfloat16_t>(gQ, sQ, C_Q);

			//ensure all smem cp finish
			__syncthreads();

			// compute S
			{
				Tensor tSrQ = thr_mmaS.partition_fragment_A(sQ); // (MMA, MMA_Br,MMA_Hc)
				Tensor tSrK = thr_mmaS.partition_fragment_B(sK); // (MMA, MMA_Bc,MMA_Hc)
				clear(tSrS);
				ldsm_and_gemm<true,true>(tSrS,tSrQ,tSrK,sQ,sK,mmaS);

				bwd_thread_print_tensor_verbose(tSrQ);
				bwd_thread_print_tensor_verbose(tSrK);
				bwd_thread_print_tensor_verbose(tSrS);
			}
			
			// load gmem L to smemL

			Tensor gL = local_tile(block_L,make_tile(Int<Br>{}),make_coord(tr));
			Tensor sL = make_tensor(make_smem_ptr(smemL),make_shape(Int<Br>{}));
			cp_D_g2s(gL,sL,C_L);
			// must wait smem ready !!!
			__syncthreads();
			bwd_thread_print_tensor_verbose(gL);
			bwd_thread_print_tensor_verbose(sL);


			

			Tensor local_p_identtiy_coord_tensor = make_identity_tensor(make_shape(Int<Br>{},Int<Bc>{}));
			Tensor local_p_identity_coord_frag = thr_mmaS.partition_C(local_p_identtiy_coord_tensor);

			bwd_thread_print_tensor_verbose(local_p_identity_coord_frag);

			Tensor cta_s_identity_coord_tensor = local_tile(s_identity_coord_tensor,make_tile(Int<Br>{},Int<Bc>{}),make_coord(tr,tc));
			Tensor s_identity_coord_frag = thr_mmaS.partition_C(cta_s_identity_coord_tensor);
			using Softmax = OnlineSoftmax<Hc, 0>;
			// compute P
			{
				bwd_thread_print_tensor_verbose(cta_s_identity_coord_tensor);
				bwd_thread_print_tensor_verbose(s_identity_coord_frag);
				for(int i = 0 ; i < size(tSrS) ; ++i){
					auto [x,y] = local_p_identity_coord_frag(i);
					if constexpr(Config::Attention  == AttentionType::Causal){
						auto [m_i,m_j] = s_identity_coord_frag(i);
						tSrS(i) =min(1.f , m_j > m_i ? 0.f : exp2f(Softmax::klog2_scale * tSrS(i) - Softmax::klog_2e * sL(x)));
					}else{
						tSrS(i) = min(1.f, exp2f(Softmax::klog_2e  * (Softmax::kscale * tSrS(i) - sL(x))));
					}
				}
			}
			// now tSrS = "P"
			bwd_thread_print_tensor_verbose(tSrS);
			
			Tensor gdO = local_tile(block_dO,make_tile(Int<Br>{},Int<Hc>{}),make_coord(tr,0));

			Tensor sdO = make_tensor(make_smem_ptr(smemdO),make_layout(make_shape(Int<Br>{},Int<Hc>{}),LayoutRight()));
			//dOcopy
			vec_cp_g2s_to_half<cute::bfloat16_t>(gdO, sdO, C_dO);
			// wait dO
			__syncthreads();
			

			// compute dP
			{
				Tensor tdPrdO = thr_mmadP.partition_fragment_A(sdO); // (MMA, MMA_Br,MMA_Hc)
				Tensor tdPrV = thr_mmadP.partition_fragment_B(sV); // (MMA, MMA_Bc,MMA_Hc)
				bwd_thread_print_tensor_verbose(sdO);
				bwd_thread_print_tensor_verbose(sV);
				// need to be clear
				clear(tdPrdP);
				ldsm_and_gemm<true, true>(tdPrdP, tdPrdO, tdPrV, sdO, sV, mmadP);

				bwd_thread_print_tensor_verbose(tdPrdP);
			}

			// load gmem D to smemD
			Tensor gD = local_tile(block_D,make_tile(Br),make_coord(tr));
			Tensor sD = make_tensor(make_smem_ptr(smemD),make_shape(Int<Br>{}));
			cp_D_g2s(gD, sD, C_D);
			// must wait smem ready !!!
			__syncthreads();
			bwd_thread_print_tensor_verbose(gD);
			bwd_thread_print_tensor_verbose(sD);

			// compute dS store to dP 
			{
				for(int i = 0 ; i < size(tdPrdP) ; ++i){
					auto [x,y] = local_p_identity_coord_frag(i);
					tdPrdP(i) = Softmax::kscale * tSrS(i) * (tdPrdP(i) - sD(x));
				}
			}
			
			bwd_thread_print_tensor_verbose(tdPrdP);

			//first convert to fp16
			Tensor tPrP = convert_type<cute::bfloat16_t>(tSrS);
			bwd_thread_print_tensor_verbose(tPrP);
			
			
			//copy P to smem for P^T
			Layout sP_layout = make_layout(make_shape(Int<Br>{},Int<Bc>{}),LayoutLeft{});
			Tensor sP = make_tensor(make_smem_ptr(smemP),sP_layout);
			cp_Cfrag_r2s(tPrP,sP,mmaS);
			// !!!
			__syncthreads();
			bwd_thread_print_tensor_verbose(sP);
			//compute dV 
			{
				Layout sPt_layout = make_layout(make_shape(Int<Bc>{},Int<Br>{}),LayoutRight{});
				Tensor sPt = make_tensor(sP.data(),sPt_layout);

				Layout sdO_reshape_layout = make_layout(make_shape(Int<Hc>{},Int<Br>{}),LayoutLeft{});
				Tensor sdO_reshape = make_tensor(sdO.data(),sdO_reshape_layout); 

				Tensor tdVrPt = thr_mmadV.partition_fragment_A(sPt);
				Tensor tdVrdO = thr_mmadV.partition_fragment_B(sdO_reshape);

				bwd_thread_print_tensor_verbose(sPt);
				bwd_thread_print_tensor_verbose(sdO_reshape);


				ldsm_and_gemm<true, false>(tdVrdV, tdVrPt, tdVrdO, sPt, sdO_reshape, mmadV);
				bwd_thread_print_tensor_verbose(tdVrPt);
				bwd_thread_print_tensor_verbose(tdVrdO);
			}
			bwd_thread_print_tensor_verbose(tdVrdV);


			//convert to fp16 for dS
			Tensor tdSrdS = convert_type<cute::bfloat16_t>(tdPrdP);
			

			// will reuse smem ,must wait all threads gemm op finish
			__syncthreads();
			

			//reuse P smem to store dS for dK after
			Layout sdS_layout = make_layout(make_shape(Int<Br>{},Int<Bc>{}),LayoutLeft{});
			Tensor sdS = make_tensor(make_smem_ptr(smemP),sdS_layout);
			cp_Cfrag_r2s(tdSrdS, sdS, mmaS);


			//reuse dO smem to store dQ
			Layout sdQ_layout = make_layout(make_shape(Int<Br>{},Int<Hc>{}),LayoutRight{});
			Tensor sdQ = make_tensor(make_smem_ptr(smemdO),sdQ_layout);

			//copy gmem dQ to smem dQ
			Tensor gdQ = local_tile(block_dQ,make_tile(Int<Br>{},Int<Hc>{}),make_coord(tr,0));
			{
				ThrCopy thr_copy_dQ = C_dQ.get_slice(threadIdx.x);
				Tensor tgdQ = thr_copy_dQ.partition_S(gdQ);
				Tensor tsdQ = thr_copy_dQ.partition_D(sdQ);
				copy(C_dQ,tgdQ,tsdQ);
			}
			
			__syncthreads();


			bwd_thread_print_tensor_verbose(gdQ);
			bwd_thread_print_tensor_verbose(sdQ);

		

			ThrMMA thr_mmadQ = mmadQ.get_slice(threadIdx.x);

			Tensor tdQrdQ = thr_mmadQ.partition_fragment_C(sdQ);

			// copy smem dQ to reg dQ
			// bwd_thread_print_tensor_verbose(gdQ);
			// bwd_thread_print_tensor_verbose(sdQ);
			{
				TiledCopy s2r_tiled_copy_dQ = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy,typename Config::TdQ>{}, mmadQ);
				ThrCopy thr_copy_dQ = s2r_tiled_copy_dQ.get_slice(threadIdx.x);
				Tensor tXsdQ = thr_copy_dQ.partition_S(sdQ);
				Tensor tXrdQ = thr_copy_dQ.retile_D(tdQrdQ);
				// bwd_thread_print_tensor_verbose(tXsdQ);
				// bwd_thread_print_tensor_verbose(tXrdQ);
				
				copy(s2r_tiled_copy_dQ,tXsdQ,tXrdQ);
			}
			
			// compute dQ
			{
				// convert dP (dP = dS) to dS as mma's A 
				Tensor tdQrdS = convert_C_frag_to_A_frag(tdSrdS);
				bwd_thread_print_tensor_verbose(tdQrdS);
				Layout sK_reshape_layout = make_layout(make_shape(Int<Hc>{},Int<Bc>{}),LayoutLeft{});
				Tensor sK_reshape = make_tensor(sK.data(), sK_reshape_layout);
				Tensor tdQrK = thr_mmadQ.partition_fragment_B(sK_reshape);
				ldsm_B_and_gemm<false>(tdQrdQ,tdQrdS,tdQrK,sK_reshape,mmadQ);
			}
			bwd_thread_print_tensor_verbose(tdQrdQ);

			// write back to sdQ
			{
				TiledCopy r2s_tiled_copy_dQ = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy,typename Config::TdQ>{}, mmadQ);
				ThrCopy thr_copy_dQ = r2s_tiled_copy_dQ.get_slice(threadIdx.x);
				Tensor tXrdQ = thr_copy_dQ.retile_S(tdQrdQ);
				Tensor tXsdQ = thr_copy_dQ.partition_D(sdQ);
				
				copy(r2s_tiled_copy_dQ,tXrdQ,tXsdQ);
			}
			//write back to gdQ
			__syncthreads();
			{
				ThrCopy thr_copy_dQ = C_dQ.get_slice(threadIdx.x);
				Tensor tsdQ = thr_copy_dQ.partition_S(sdQ);
				Tensor tgdQ = thr_copy_dQ.partition_D(gdQ);
				copy(C_dQ,tsdQ,tgdQ);
				bwd_thread_print_tensor_verbose(tgdQ);
			}
			bwd_thread_print_tensor_verbose(gdQ);



			bwd_thread_print_tensor_verbose(tdSrdS);
			bwd_thread_print_tensor_verbose(sdS);
			
			//wait sdS finish
			__syncthreads();
			//compute dK
			{
				Tensor sdSt = make_tensor(sdS.data(),make_layout(make_shape(Int<Bc>{},Int<Br>{}),LayoutRight{})); 
				Tensor tdKrdSt = thr_mmadK.partition_fragment_A(sdSt);
				Tensor sQ_reshape = make_tensor(sQ.data(),make_layout(make_shape(Int<Hc>{},Int<Br>{}),LayoutLeft{}));
				Tensor tdKrQ = thr_mmadK.partition_fragment_B(sQ_reshape);
				ldsm_and_gemm<true,false>(tdKrdK, tdKrdSt, tdKrQ, sdSt,sQ_reshape, mmadK);
			}
			bwd_thread_print_tensor_verbose(tdKrdK);

		}

		cp_Cfrag_r2s(tdKrdK, sdK, mmadK);
		cp_Cfrag_r2s(tdVrdV, sdV, mmadV);

		Tensor gdK =local_tile(block_dK,make_tile(Int<Bc>{},Int<Hc>{}),make_coord(tc,0));
		Tensor gdV =local_tile(block_dV,make_tile(Int<Bc>{},Int<Hc>{}),make_coord(tc,0));

		__syncthreads();

		vec_cp_s2g(sdK, gdK, C_dK);
		vec_cp_s2g(sdV, gdV, C_dV);

		{
			// __syncthreads();
			bwd_thread_print_tensor_verbose(gdK);
			bwd_thread_print_tensor_verbose(gdV);
		}

	}
}
