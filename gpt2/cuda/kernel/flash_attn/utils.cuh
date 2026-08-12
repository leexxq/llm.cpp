#pragma once
#include <cute/tensor.hpp>
#include "cute/numeric/numeric_types.hpp"
#include "cutlass/numeric_conversion.h"
using namespace cute;
#define FORCE_INLINE __forceinline__
// #define FORCE_INLINE 

template <class To_type, class Engine, class Layout>
FORCE_INLINE __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
	using From_type = typename Engine::value_type;

	Tensor target = make_fragment_like<To_type>(tensor.layout());

	auto convert_op = cutlass::NumericConverter<To_type, From_type>{};

	CUTE_UNROLL
	for (int i = 0; i < size(tensor); ++i) {
		target(i) = convert_op(tensor(i));
	}
	return target;
}



template<class Engine , class Layout>
FORCE_INLINE __device__ auto convert_C_frag_to_A_frag(Tensor<Engine,Layout>  & tensor){
			// convert s frag to a frag for o
	auto retile_shape = make_shape(make_shape(size<0, 0>(tensor), size<0, 1>(tensor), _2{}), size<1>(tensor), size<2>(tensor) / _2{});
	auto retile_stride = make_stride(make_stride(stride<0, 0>(tensor), stride<0, 1>(tensor), stride<2>(tensor)), stride<1>(tensor), stride<2>(tensor) * _2{});

	return make_tensor(tensor.data(), make_layout(retile_shape, retile_stride));
}


// Here, "row" means that I place a row of data owned by a thread at the first index,
// while the second index corresponds to other rows.
template <class Engine,class Layout>
FORCE_INLINE __device__ auto convert_C_to_row(Tensor<Engine,Layout>  & ctensor) {
	// retile the register file layout (due per thread ownning multi row datas, we each calculate them.)

	auto row_tiler = make_layout(
			make_shape(size<0, 0>(ctensor), Int<1>{}, size<2>(ctensor)),
			make_stride(Int<1>{}, Int<0>{}, size<0>(ctensor) * size<1>(ctensor)));
	return make_tensor(ctensor.data(),zipped_divide(ctensor.layout(), row_tiler)); //(nums_pre_row,rows)
}




template<class STensor>
FORCE_INLINE __device__ auto convert_to_row_for_D(STensor& tensor){
	Layout flat_layout = flatten(tensor.layout());//(x,y, M,N)
	auto row_layout = append(select<1,3>(flat_layout),select<0,2>(flat_layout));
	return group_modes<0,2>(make_tensor(tensor.data(),row_layout));
}
template<class STensor , class GTensor , class TiledCopy>
FORCE_INLINE __device__ void vec_cp_s2g(STensor& stensor , GTensor& gtensor , TiledCopy tiled_copy){
		ThrCopy thr_copy = tiled_copy.get_slice(threadIdx.x);
		Tensor ts = thr_copy.partition_S(stensor);
		Tensor tg = thr_copy.partition_D(gtensor);
		copy(tiled_copy,ts,tg);
}


template<class T = cute::half_t , class GTensor , class STensor , class TiledCopy>
FORCE_INLINE __device__ void vec_cp_g2s_to_half(GTensor& gtensor , STensor& stensor , TiledCopy tiled_copy){
		ThrCopy thr_copy = tiled_copy.get_slice(threadIdx.x);
		Tensor tg = thr_copy.partition_S(gtensor);
		Tensor ts = thr_copy.partition_D(stensor);
		Tensor tr = thr_copy.retile_D(make_fragment_like(ts));
		copy(tiled_copy,tg,tr);
		Tensor tr_fp16  = convert_type<T>(tr);
		copy(tiled_copy,tr_fp16,ts);
}



// only ldsm B to b_frag then gemm A*B
template<bool BRowMajor, class CFrag,class AFrag,class BFrag,class BSTensor,class MMA>
FORCE_INLINE __device__ void ldsm_B_and_gemm(CFrag& c_frag,AFrag& a_frag,BFrag& b_frag,BSTensor& b_stensor,MMA mma){
	ThrMMA thr_mmaS  = mma.get_slice(threadIdx.x);


	using CopyAtomB = std::conditional_t<BRowMajor,Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t> , Copy_Atom<SM75_U16x8_LDSM_T, cute::half_t>>;
	CopyAtomB s2r_atom_B;
	TiledCopy s2r_copy_B = make_tiled_copy_B(s2r_atom_B, mma);
	ThrCopy s2r_thr_copy_B = s2r_copy_B.get_slice(threadIdx.x);

	Tensor tXsB = s2r_thr_copy_B.partition_S(b_stensor); //(CPY,MMA_T,MMA_Hc)
	Tensor tXrB = s2r_thr_copy_B.retile_D(b_frag);


	copy(s2r_atom_B, tXsB(_,_,_0{}), tXrB(_,_,_0{}));

	CUTE_UNROLL
	for (int k = 0; k < size<2>(a_frag); ++k) {
		gemm(mma, a_frag(_, _, k), b_frag(_, _, k),c_frag);
		if( k < size<2>(tXsB) - _1{}){
			copy(s2r_atom_B, tXsB(_,_,k+_1{}), tXrB(_,_,k+_1{}));
		}
	}
}


// ldsm A and B then accumlate to c_frag
// We consider the shapes of matrices A and B in the matrix multiplication to be (m, k) and (n, k), 
// where k is the intermediate dimension.
template<bool ARowMajor, bool BRowMajor, class CFrag,class AFrag,class BFrag,class ASTensor,class BSTensor,class MMA>
FORCE_INLINE __device__ void ldsm_and_gemm(CFrag& c_frag,AFrag& a_frag,BFrag& b_frag,ASTensor& a_stensor,BSTensor& b_stensor,MMA mma){
	ThrMMA thr_mmaS  = mma.get_slice(threadIdx.x);

	using CopyAtomA = std::conditional_t<ARowMajor,Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t> , Copy_Atom<SM75_U16x8_LDSM_T, cute::half_t>>;
	CopyAtomA s2r_atom_A;
	
	TiledCopy s2r_copy_A = make_tiled_copy_A(s2r_atom_A, mma);
	ThrCopy s2r_thr_copy_A = s2r_copy_A.get_slice(threadIdx.x);

	Tensor tXsA = s2r_thr_copy_A.partition_S(a_stensor);
	Tensor tXrA = s2r_thr_copy_A.retile_D(a_frag);


	using CopyAtomB = std::conditional_t<BRowMajor,Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t> , Copy_Atom<SM75_U16x8_LDSM_T, cute::half_t>>;
	CopyAtomB s2r_atom_B;
	TiledCopy s2r_copy_B = make_tiled_copy_B(s2r_atom_B, mma);
	ThrCopy s2r_thr_copy_B = s2r_copy_B.get_slice(threadIdx.x);

	Tensor tXsB = s2r_thr_copy_B.partition_S(b_stensor); //(CPY,MMA_T,MMA_Hc)
	Tensor tXrB = s2r_thr_copy_B.retile_D(b_frag);


	copy(s2r_atom_A, tXsA(_,_,_0{}), tXrA(_,_,_0{}));
	copy(s2r_atom_B, tXsB(_,_,_0{}), tXrB(_,_,_0{}));

	CUTE_UNROLL
	for (int k = 0; k < size<2>(a_frag); ++k) {
		gemm(mma, a_frag(_, _, k), b_frag(_, _, k),c_frag);
		if( k < size<2>(tXsA) - _1{}){
			copy(s2r_atom_A, tXsA(_,_,k+_1{}), tXrA(_,_,k+_1{}));
			copy(s2r_atom_B, tXsB(_,_,k+_1{}), tXrB(_,_,k+_1{}));
		}
	}
}


template<class GTensor , class STensor , class TiledCopy>
FORCE_INLINE __device__ void cp_D_g2s(GTensor& gtensor , STensor& stensor,TiledCopy tiled_copy){

	ThrCopy thr_copy = tiled_copy.get_slice(threadIdx.x);
	Tensor tg = thr_copy.partition_S(gtensor);
	Tensor ts = thr_copy.partition_D(stensor);
	Tensor tc = make_identity_tensor(make_shape(size(gtensor)));
	Tensor tc_frag = thr_copy.partition_D(tc);
	
	for(int i = 0 ; i < size<1>(tc_frag); ++i){
		if(elem_less(get<0>(tc_frag(0,i)),size(tc))){
			copy(tiled_copy,tg,ts);
		}
	}

}

template<class RTensor,class STensor,class MMA>
FORCE_INLINE __device__  void cp_Cfrag_r2s(RTensor& rtensor,STensor& stensor,MMA mma){
	TiledCopy tiled_copy = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy,cute::half_t>{},mma);
	ThrCopy thr_copy = tiled_copy.get_slice(threadIdx.x);
	Tensor ts = thr_copy.partition_D(stensor);
	// S is P 
	Tensor tr = thr_copy.retile_S(rtensor);
	copy(tiled_copy,tr,ts);

}

template<class RTensor,class GTensor,class MMA>
FORCE_INLINE __device__  void cp_r2g_C(RTensor& rtensor,GTensor& gtensor,MMA mma){
	cp_Cfrag_r2s(rtensor, gtensor, mma);
}

