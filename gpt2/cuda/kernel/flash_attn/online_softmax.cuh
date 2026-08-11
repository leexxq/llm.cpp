#pragma once
#include <cute/tensor.hpp>
#include "config.cuh"
#include "math_constants.h"
using namespace cute;
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
		CUTE_STATIC_ASSERT_V(size<1>(row_tOrO) == size<1>(row_identity_coord));


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

	template<class L_tensor, class M_tensor>
	__forceinline__ __device__ void write_logsumexp(L_tensor& ltensor , const M_tensor& row_identity_coord) {
		if(threadIdx.x % 4 == 0){
			CUTE_UNROLL
			for(int r = 0 ; r < size<1>(row_identity_coord); ++r){
				auto [i,j] = row_identity_coord(0,r);
				ltensor(i) = tmaxs_frag(r) * kscale + logf(tsums_frag(r));
			}
		}
	}


};