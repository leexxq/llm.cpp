#include "layer.h"
#include "log.h"
#include "global.h"

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <cassert>
#include <cstdio>

void Layer::Init(size_t B, size_t T, size_t C, size_t V, size_t NH) {
	layernorm1 = LayerNorm{ B, T, C };
	qkv = MatMul{ C, 3 * C };
	attention = Attention{ B, T, C, NH };
	att_proj = MatMul{ C, C };
	residual1 = Residual{ B, T, C };
	layernorm2 = LayerNorm{ B, T, C };
	fch = MatMul{ C, 4 * C };
	fch_gelu = GELU{};
	fcproj = MatMul{ 4 * C, C };
	residual2 = Residual{ B, T, C };

	dl_residual2 = makeZero(B, T, C);
	dl_fcproj = makeZero(B, T, C);
	dl_fch_gelu = makeZero(B, T, 4 * C);
	dl_fch = makeZero(B, T, 4 * C);
	dl_ln2 = makeZero(B, T, C);
	dresidual = makeZero(B, T, C);
	dl_attproj = makeZero(B, T, C);
	dl_atty = makeZero(B, T, C);
	dl_qkv = makeZero(B, T, 3 * C);
	dl_ln1 = makeZero(B, T, C);
}

VecBTC Layer::Forward(const VecBTC &residual) {
	l_ln1_ = layernorm1.Forward(residual);
	DEBUG_PRINTLN("layernorm forward sharp: ({} , {} , {})", l_ln1_.size(), l_ln1_[0].rows(), l_ln1_[0].cols());
	DEBUG_PRINTLN("l_ln1[0]: \n{}", fmt::streamed(l_ln1_[0].block<2, 2>(0, 0)));

	l_qkv_ = qkv.Forward(l_ln1_);
	DEBUG_PRINTLN("qkv_ forward sharp: ({} , {} , {})", l_qkv_.size(), l_qkv_[0].rows(), l_qkv_[0].cols());
	DEBUG_PRINTLN("l_qkv[0]: \n{}", fmt::streamed(l_qkv_[0].block<2, 2>(0, 0)));

	l_atty_ = attention.Forward(l_qkv_);
	DEBUG_PRINTLN("attention forward sharp: ({} , {} , {})", l_atty_.size(), l_atty_[0].rows(), l_atty_[0].cols());
	DEBUG_PRINTLN("l_atty[0]: \n{}", fmt::streamed(l_atty_[0].block<2, 2>(0, 0)));
	DEBUG_PRINTLN("\t att's shape is: ({} , {} , {} , {})", attention.att.size(), attention.att.front().size(), attention.att.front().front().rows(), attention.att.front().front().cols());
	DEBUG_PRINTLN("\t att[0][0]: \n {}", fmt::streamed(attention.att[0][0].block<8, 8>(0, 0)));
	DEBUG_PRINTLN("\t preatt's shape is: ({} , {} , {} , {})", attention.pre_att.size(), attention.pre_att.front().size(), attention.pre_att.front().front().rows(), attention.pre_att.front().front().cols());
	DEBUG_PRINTLN("\t preatt[0][0]: \n {}", fmt::streamed(attention.pre_att[0][0].block<8, 8>(0, 0)));

	l_attproj_ = att_proj.Forward(l_atty_);
	DEBUG_PRINTLN("att_proj forward sharp: ({} , {} , {})", l_attproj_.size(), l_attproj_[0].rows(), l_attproj_[0].cols());
	DEBUG_PRINTLN("l_attproj[0]: \n{}", fmt::streamed(l_attproj_[0].block<2, 2>(0, 0)));

	l_residual2_ = residual1.Forward(residual, l_attproj_);
	DEBUG_PRINTLN("l_residual2 forward sharp: ({} , {} , {})", l_residual2_.size(), l_residual2_[0].rows(), l_residual2_[0].cols());
	DEBUG_PRINTLN("l_residual2[0]: \n{}", fmt::streamed(l_residual2_[0].block<2, 2>(0, 0)));

	l_ln2_ = layernorm2.Forward(l_residual2_);
	DEBUG_PRINTLN("l_ln2 forward sharp: ({} , {} , {})", l_ln2_.size(), l_ln2_[0].rows(), l_ln2_[0].cols());
	DEBUG_PRINTLN("l_ln2[0]: \n{}", fmt::streamed(l_ln2_[0].block<2, 2>(0, 0)));

	l_fch_ = fch.Forward(l_ln2_);
	DEBUG_PRINTLN("l_fch forward sharp: ({} , {} , {})", l_fch_.size(), l_fch_[0].rows(), l_fch_[0].cols());
	DEBUG_PRINTLN("l_fch[0]: \n{}", fmt::streamed(l_fch_[0].block<2, 2>(0, 0)));

	l_fch_gelu_ = fch_gelu.Forward(l_fch_);
	DEBUG_PRINTLN("l_fch_gelu forward sharp: ({} , {} , {})", l_fch_gelu_.size(), l_fch_gelu_[0].rows(), l_fch_gelu_[0].cols());
	DEBUG_PRINTLN("l_fch_gelu[0]: \n{}", fmt::streamed(l_fch_gelu_[0].block<2, 2>(0, 0)));

	l_fcproj_ = fcproj.Forward(l_fch_gelu_);
	DEBUG_PRINTLN("l_fcproj forward sharp: ({} , {} , {})", l_fcproj_.size(), l_fcproj_[0].rows(), l_fcproj_[0].cols());
	DEBUG_PRINTLN("l_fcproj[0]: \n{}", fmt::streamed(l_fcproj_[0].block<2, 2>(0, 0)));

	l_residual3_ = residual2.Forward(l_residual2_, l_fcproj_);
	DEBUG_PRINTLN("l_residual3 forward sharp: ({} , {} , {})", l_residual3_.size(), l_residual3_[0].rows(), l_residual3_[0].cols());
	DEBUG_PRINTLN("l_residual3[0]: \n{}", fmt::streamed(l_residual3_[0].block<2, 2>(0, 0)));
	return l_residual3_;
}

VecBTC Layer::Backward(const VecBTC &d_outputs, const VecBTC &residual) {
	auto dresidual2_wrapper = residual2.Backward(d_outputs);
	dl_residual2 += dresidual2_wrapper.first;
	dl_fcproj += dresidual2_wrapper.second;

	assert(dl_residual2.size() > 0);
	DEBUG_PRINTLN("residual2_ backward sharp: ({} , {} , {})", dl_residual2.size(), dl_residual2[0].rows(), dl_residual2[0].cols());
	DEBUG_PRINTLN("dl_residual2_[0]: \n{}", fmt::streamed(dl_residual2[0].block<2, 2>(0, 0)));
	DEBUG_PRINTLN("dl_fcproj[0]: \n{}", fmt::streamed(dl_fcproj[0].block<2, 2>(0, 0)));

	dl_fch_gelu += fcproj.Backward(dl_fcproj, l_fch_gelu_);
	assert(dl_fch_gelu.size() > 0);
	DEBUG_PRINTLN("fch_gelu_ backward sharp: ({} , {} , {})", dl_fch_gelu.size(), dl_fch_gelu[0].rows(), dl_fch_gelu[0].cols());
	DEBUG_PRINTLN("dl_fch_gelu[0]: \n{}", fmt::streamed(dl_fch_gelu[0].block<2, 2>(0, 0)));

	dl_fch += fch_gelu.Backward(dl_fch_gelu, l_fch_);
	assert(dl_fch.size() > 0);
	DEBUG_PRINTLN("fch_gelu_ backward sharp: ({} , {} , {})", dl_fch.size(), dl_fch[0].rows(), dl_fch[0].cols());
	DEBUG_PRINTLN("dl_fch[0]: \n{}", fmt::streamed(dl_fch[0].block<2, 2>(0, 0)));

	dl_ln2 += fch.Backward(dl_fch, l_ln2_);
	assert(dl_ln2.size() > 0);
	DEBUG_PRINTLN("fch_ backward sharp: ({} , {} , {})", dl_ln2.size(), dl_ln2[0].rows(), dl_ln2[0].cols());
	DEBUG_PRINTLN("dl_ln2[0]: \n{}", fmt::streamed(dl_ln2[0].block<2, 2>(0, 0)));

	dl_residual2 += layernorm2.Backward(dl_ln2, l_residual2_);
	assert(dl_residual2.size() > 0);
	DEBUG_PRINTLN("layernorm2_ backward sharp: ({} , {} , {})", dl_residual2.size(), dl_residual2[0].rows(), dl_residual2[0].cols());
	DEBUG_PRINTLN("dl_residual2_[0]: \n{}", fmt::streamed(dl_residual2[0].block<2, 2>(0, 0)));

	auto dresidual1_wrap = residual1.Backward(dl_residual2);
	dl_attproj += dresidual1_wrap.first;

	assert(dl_attproj.size() > 0);
	DEBUG_PRINTLN("residual1_ backward sharp: ({} , {} , {})", dl_attproj.size(), dl_attproj[0].rows(), dl_attproj[0].cols());
	assert(dl_residual2.size() > 0);
	DEBUG_PRINTLN("dl_attproj[0]: \n{}", fmt::streamed(dl_attproj[0].block<2, 2>(0, 0)));
	DEBUG_PRINTLN("dl_residual2[0]: \n{}", fmt::streamed(dl_residual2[0].block<2, 2>(0, 0)));

	dl_atty += att_proj.Backward(dl_attproj, l_atty_);
	assert(dl_atty.size() > 0);
	DEBUG_PRINTLN("att_proj_ backward sharp: ({} , {} , {})", dl_atty.size(), dl_atty[0].rows(), dl_atty[0].cols());
	DEBUG_PRINTLN("dl_atty[0]: \n{}", fmt::streamed(dl_atty[0].block<2, 2>(0, 0)));

	dl_qkv += attention.Backward(dl_atty, l_qkv_);
	assert(dl_qkv.size() > 0);
	DEBUG_PRINTLN("attention_ backward sharp: ({} , {} , {})", dl_qkv.size(), dl_qkv[0].rows(), dl_qkv[0].cols());
	DEBUG_PRINTLN("dl_qkv[0]: \n{}", fmt::streamed(dl_qkv[0].block<8, 8>(0, 0)));

	dl_ln1 += qkv.Backward(dl_qkv, l_ln1_);
	assert(dl_ln1.size() > 0);
	DEBUG_PRINTLN("qkv_ backward sharp: ({} , {} , {})", dl_ln1.size(), l_ln1_[0].rows(), l_ln1_[0].cols());
	DEBUG_PRINTLN("dl_ln1_[0]: \n{}", fmt::streamed(dl_ln1[0].block<2, 2>(0, 0)));

	dresidual1_wrap.second += layernorm1.Backward(dl_ln1, residual);

	assert(dresidual1_wrap.second.size() > 0);
	DEBUG_PRINTLN("layernorm backward sharp: ({} , {} , {})", dresidual1_wrap.second.size(), dresidual1_wrap.second[0].rows(), dresidual1_wrap.second[0].cols());
	DEBUG_PRINTLN("dresidual1_wrap.second[0]: \n{}", fmt::streamed(dresidual1_wrap.second[0].block<2, 2>(0, 0)));

	return dresidual1_wrap.second;
}
