#pragma once
#include "cute/layout_composed.hpp"
#include "cute/swizzle.hpp"
#include <cute/tensor.hpp>
using namespace cute;
enum class AttentionType{
	Default,
	Causal,
};


template <
int Br,int Bc,int Hc,
class TL , class TD,
class TdQ ,class TdK ,class TdV,
class TMMA = cute::half_t
>
								
struct FlashBwdSharedStorage
{
	cute::ArrayEngine<TMMA, Int<Br>{} * Int<Hc>{}> Q;
	cute::ArrayEngine<TMMA, Int<Bc>{} * Int<Hc>{}> K;
	cute::ArrayEngine<TMMA, Int<Bc>{} * Int<Hc>{}> V;

	cute::ArrayEngine<TMMA, Int<Br>{} * Int<Bc>{}> P;
	cute::ArrayEngine<TMMA, Int<Br>{} * Int<Hc>{}> dO;

	cute::ArrayEngine<TL, Int<Br>{}> L;
	cute::ArrayEngine<TD, Int<Br>{}> D;

	cute::ArrayEngine<TdK, Int<Bc>{} * Int<Hc>{}> dK;
	cute::ArrayEngine<TdV, Int<Bc>{} * Int<Hc>{}> dV;

};

template <int HeadDim,AttentionType Att>
struct FlashBwdConfigFp32{
	static constexpr int kthreads = 128;
	static constexpr int kval_per_thread = 4;
	static constexpr int kthreads_per_row =32 * 4 / (sizeof(float) * kval_per_thread);
	static constexpr int kthreads_per_col = kthreads / kthreads_per_row;

	static constexpr int kBr = 64;
    static constexpr int kBc = 64;
	static constexpr int kHc = HeadDim;

	
	// using SharedStorage = FlashBwdSharedStorage<kBr,kBc,kHc,float,float,float,float,float,cute::half_t>;
	using SharedStorage = FlashBwdSharedStorage<kBr,kBc,kHc,float,float,float,float,float,cute::bfloat16_t>;

	static constexpr int smem_size = int(sizeof(SharedStorage)); 

	static constexpr AttentionType Attention = Att;

	// using MMAOP = SM80_16x8x16_F32F16F16F32_TN;
	using MMAOP = SM80_16x8x16_F32BF16BF16F32_TN;
	using MMALayout = Layout<Shape<_4, _1, _1>>;
	using MMATile_S = Tile<Int<kBr>, Int<kBc>, _16>;
	using MMATile_dQ = Tile<Int<kBr>, Int<kHc>, _16>;
	using MMATile_dK = Tile<Int<kBc>, Int<kHc>, _16>;
	using MMA_S = decltype(make_tiled_mma(MMAOP{}, MMALayout{}, MMATile_S{}));
	using MMA_dQ = decltype(make_tiled_mma(MMAOP{}, MMALayout{}, MMATile_dQ{}));
	using MMA_dK = decltype(make_tiled_mma(MMAOP{}, MMALayout{}, MMATile_dK{}));


	using VecCpyAtom = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>,float>;

	
	using UniversalTiledCopy = decltype(make_tiled_copy(VecCpyAtom{},
		Layout<Shape<Int<kthreads_per_col>, Int<kthreads_per_row>>, Stride<Int<kthreads_per_row>, _1>>{},
		Layout<Shape<_1, Int<kval_per_thread>>>{}));

	using LDTiledCopy = decltype(make_tiled_copy(VecCpyAtom{},
								Layout<Shape< Int<kBr/4> >,Stride<_1>>{},
								Layout<Shape<_4>>{})); 
	using TQ = float;
	using TK = float;
	using TV = float;
	using TO = float;
	using TdO = float;
	using TD = float;
	using TL = float;
	using TdQ = float;
	using TdK = float;
	using TdV = float;

	using CopyQ = UniversalTiledCopy;
	using CopyK = UniversalTiledCopy;
	using CopyV = UniversalTiledCopy;
	using CopyO =  UniversalTiledCopy;
	using CopydO = UniversalTiledCopy;
	using CopyL = LDTiledCopy;
	using CopyD = LDTiledCopy;
	using CopydQ = UniversalTiledCopy;
	using CopydK = UniversalTiledCopy;
	using CopydV = UniversalTiledCopy;
};


template <
int Br,int Bc,int Hc,
class TQ ,class TK ,class TV>
								
struct FlashFwdSharedStorage
{
	cute::ArrayEngine<TQ, Int<Br>{} * Int<Hc>{}> Q;
	cute::ArrayEngine<TK, Int<Bc>{} * Int<Hc>{}> K;
	cute::ArrayEngine<TV, Int<Bc>{} * Int<Hc>{}> V;
};

template <int HeadDim,AttentionType Att>
struct FlashFwdConfigFp32{
	static constexpr int kthreads = 128;
	static constexpr int kval_per_thread = 4;
	static constexpr int kthreads_per_row =32 * 4 / (sizeof(float) * kval_per_thread);
	static constexpr int kthreads_per_col = kthreads / kthreads_per_row;

	static constexpr int kBr = 128;
    static constexpr int kBc = 128;
	static constexpr int kHc = HeadDim;

	
	using TSharedQ = float;
	using TSharedK = float;
	using TSharedV = cute::half_t;
	using SharedStorage = FlashFwdSharedStorage<kBr,kBc,kHc,TSharedQ,TSharedK,TSharedV>;

	using SwzV_64Atom = decltype(
		composition(Swizzle<3,3,3>{},
					Layout<Shape <_8,Shape <_8, _8>>,
					Stride<_8,Stride<_1,_64>>>{}));

	using SwzV_32Atom= decltype(
		composition(Swizzle<2,3,3>{},
					Layout<Shape <_8,Shape <_8, _4>>,
					Stride<_8,Stride<_1,_64>>>{}));

	
	using SwzV_Atom = std::conditional_t<kHc == 32 , SwzV_32Atom,SwzV_64Atom>;

 

	// using SV_Swz_eq32 = decltype(tile_to_shape(SwzV_32Atom{},make_shape(Int<kBc>{},Int<kHc>{})));

	// using SV_Swz_gr32 = decltype(tile_to_shape(SwzV_Atom{},make_shape(Int<kBc>{},Int<kHc>{})));

	using SV_Swz = decltype(tile_to_shape(SwzV_Atom{},make_shape(Int<kBc>{},Int<kHc>{})));

	using SwzQK_Atom = decltype(
		composition(Swizzle<3,2,3>{},
					Layout<Shape <_4,Shape <_4, _8>>,
					Stride<_4,Stride<_1,_16>>>{}));

	using SQ_Swz = decltype(tile_to_shape(SwzQK_Atom{},make_shape(Int<kBr>{},Int<kHc>{})));
	using SK_Swz = decltype(tile_to_shape(SwzQK_Atom{},make_shape(Int<kBc>{},Int<kHc>{})));


	static constexpr int smem_size = int(sizeof(SharedStorage)); 

	static constexpr AttentionType Attention = Att;

	// using MMAOP = SM80_16x8x16_F32F16F16F32_TN;
	using MMAS = decltype(make_tiled_mma(SM80_16x8x8_F32TF32TF32F32_TN{}, Layout<Shape<_4, _1, _1>>{}, Tile<Int<kBr>, Int<kBc>, _8>{}));

	using MMAO = decltype(make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<_4, _1, _1>>{}, Tile<Int<kBr>, Int<kHc>, _16>{}));


	


	using TQ = float;
	using TK = float;
	using TV = float;
	using TO = float;
	using TL = float;
	using AsyncVecCPAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cutlass::uint128_t>,float>;
	using VecCpyAtom = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>,float>;

	using CPThrLayout = 
		Layout<Shape<Int<kthreads_per_col>, Int<kthreads_per_row>>, Stride<Int<kthreads_per_row>, _1>>;
	using CPValLayout = 
		Layout<Shape<_1, Int<kval_per_thread>>>;

	using UniversalTiledCopy = decltype(make_tiled_copy(VecCpyAtom{},
		CPThrLayout{},
		CPValLayout{}));

	using CopyQ = decltype(make_tiled_copy(AsyncVecCPAtom{},CPThrLayout{},CPValLayout{}));
	using CopyK = CopyQ;
	using CopyV = UniversalTiledCopy;
	using CopyO = UniversalTiledCopy;
};