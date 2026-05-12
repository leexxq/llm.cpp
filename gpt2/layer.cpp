#include "layer.h"

#include "global.h"

#include <fmt/core.h>
#include <fmt/ostream.h>

void Layer::Init(size_t B, size_t T, size_t C, size_t V, size_t NH) {
	layernorm1_ = LayerNorm{ B, T, C };
	qkv_ = MatMul{ C, 3 * C };
	attention_ = Attention{ B, T, C, NH };
	att_proj_ = MatMul{ C, C };
	residual1_ = Residual{};
	layernorm2_ = LayerNorm{ V, T, C };
	fch_ = MatMul{ C, 4 * C };
	fch_gelu_ = GELU{};
	fcproj_ = MatMul{ 4 * C, C };
	residual2_ = Residual{};
}

VecBTC Layer::Forward(const VecBTC &residual) {
	VecBTC l_ln1 = layernorm1_.Forward(residual);
	DEBUG_PRINTLN("layernorm forward sharp: ({} , {} , {})", l_ln1.size(), l_ln1[0].rows(), l_ln1[0].cols());
	DEBUG_PRINTLN("l_ln1[0]: \n{}", fmt::streamed(l_ln1[0].block<2, 2>(1, 1)));

	// 3C is GPT2's dim before qkv split
	using VecBT3C = VecBTC;
	VecBT3C l_qkv = qkv_.Forward(l_ln1);
	DEBUG_PRINTLN("qkv_ forward sharp: ({} , {} , {})", l_qkv.size(), l_qkv[0].rows(), l_qkv[0].cols());
	DEBUG_PRINTLN("l_qkv[0]: \n{}", fmt::streamed(l_qkv[0].block<2, 2>(1, 1)));

	VecBTC l_atty = attention_.Forward(l_qkv);
	DEBUG_PRINTLN("attention forward sharp: ({} , {} , {})", l_atty.size(), l_atty[0].rows(), l_atty[0].cols());
	DEBUG_PRINTLN("l_atty[0]: \n{}", fmt::streamed(l_atty[0].block<2, 2>(1, 1)));
	DEBUG_PRINTLN("-----att's shape is: ({} , {} , {} , {})", attention_.att.size(), attention_.att.front().size(), attention_.att.front().front().rows(), attention_.att.front().front().cols());

	VecBTC l_attproj = att_proj_.Forward(l_atty);
	DEBUG_PRINTLN("att_proj forward sharp: ({} , {} , {})", l_attproj.size(), l_attproj[0].rows(), l_attproj[0].cols());
	DEBUG_PRINTLN("l_attproj[0]: \n{}", fmt::streamed(l_attproj[0].block<2, 2>(1, 1)));

	VecBTC l_residual2 = residual1_.Forward(residual, l_attproj);
	DEBUG_PRINTLN("l_residual2 forward sharp: ({} , {} , {})", l_residual2.size(), l_residual2[0].rows(), l_residual2[0].cols());
	DEBUG_PRINTLN("l_residual2[0]: \n{}", fmt::streamed(l_residual2[0].block<2, 2>(1, 1)));

	VecBTC l_ln2 = layernorm2_.Forward(l_residual2);
	DEBUG_PRINTLN("l_ln2 forward sharp: ({} , {} , {})", l_ln2.size(), l_ln2[0].rows(), l_ln2[0].cols());
	DEBUG_PRINTLN("l_ln2[0]: \n{}", fmt::streamed(l_ln2[0].block<2, 2>(1, 1)));

	using VecBT4C = VecBTC;
	VecBT4C l_fch = fch_.Forward(l_ln2);
	DEBUG_PRINTLN("l_fch forward sharp: ({} , {} , {})", l_fch.size(), l_fch[0].rows(), l_fch[0].cols());
	DEBUG_PRINTLN("l_fch[0]: \n{}", fmt::streamed(l_fch[0].block<2, 2>(1, 1)));

	VecBT4C l_fch_gelu = fch_gelu_.Forward(l_fch);
	DEBUG_PRINTLN("l_fch_gelu forward sharp: ({} , {} , {})", l_fch_gelu.size(), l_fch_gelu[0].rows(), l_fch_gelu[0].cols());
	DEBUG_PRINTLN("l_fch_gelu[0]: \n{}", fmt::streamed(l_fch_gelu[0].block<2, 2>(1, 1)));

	VecBTC l_fcproj = fcproj_.Forward(l_fch_gelu);
	DEBUG_PRINTLN("l_fcproj forward sharp: ({} , {} , {})", l_fcproj.size(), l_fcproj[0].rows(), l_fcproj[0].cols());
	DEBUG_PRINTLN("l_fcproj[0]: \n{}", fmt::streamed(l_fcproj[0].block<2, 2>(1, 1)));

	VecBTC l_residual3 = residual2_.Forward(l_residual2, l_fcproj);
	DEBUG_PRINTLN("l_residual3 forward sharp: ({} , {} , {})", l_residual3.size(), l_residual3[0].rows(), l_residual3[0].cols());
	DEBUG_PRINTLN("l_residual3[0]: \n{}", fmt::streamed(l_residual3[0].block<2, 2>(1, 1)));
	return l_residual3;
}
