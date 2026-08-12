#pragma once
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