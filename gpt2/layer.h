#pragma once

#include "attention.h"
#include "gelu.h"
#include "global.h"
#include "layernorm.h"
#include "matmul.h"
#include "residual.h"

class Layer {
	// 3C is GPT2's dim before qkv split
	using VecBT3C = VecBTC;
	using VecBT4C = VecBTC;

private:
	VecBTC l_residual3_;
	VecBTC l_fcproj_;
	VecBT4C l_fch_gelu_;
	VecBT4C l_fch_;
	VecBTC l_ln2_;
	VecBTC l_residual2_;
	VecBTC l_attproj_;
	VecBTC l_atty_;
	VecBT3C l_qkv_;
	VecBTC l_ln1_;

	VecBTC dl_residual2_;
	VecBTC dl_fcproj_;
	VecBT4C dl_fch_gelu_;
	VecBT4C dl_fch_;
	VecBTC dl_ln2_;
	VecBTC dresidual_;
	VecBTC dl_attproj_;
	VecBTC dl_residual_;
	VecBTC dl_atty_;
	VecBTC dl_qkv_;
	VecBTC dl_ln1_;

	void Init(size_t B, size_t T, size_t C, size_t V, size_t NH);

public:
	LayerNorm layernorm1;
	LayerNorm layernorm2;

	MatMul fch;
	MatMul fcproj;
	MatMul qkv;
	MatMul att_proj;
	MatMul mm_logits;

	GELU fch_gelu;

	Attention attention;

	Residual residual1;
	Residual residual2;

public:
	Layer() {}
	Layer(size_t B, size_t T, size_t C, size_t V, size_t NH) { Init(B, T, C, V, NH); }

	VecBTC Forward(const VecBTC &inputs);

	VecBTC Backward(const VecBTC &d_outputs, const VecBTC &inputs);
};
