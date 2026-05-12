#pragma once

#include "attention.h"
#include "gelu.h"
#include "global.h"
#include "layernorm.h"
#include "matmul.h"
#include "residual.h"

class Layer {
private:
	LayerNorm layernorm1_;
	LayerNorm layernorm2_;

	MatMul fch_;
	MatMul fcproj_;
	MatMul qkv_;
	MatMul att_proj_;
	MatMul mm_logits_;

	GELU fch_gelu_;

	Attention attention_;

	Residual residual1_;
	Residual residual2_;

	void Init(size_t B, size_t T, size_t C, size_t V, size_t NH);

public:
	Layer() {}
	Layer(size_t B, size_t T, size_t C, size_t V, size_t NH) { Init(B, T, C, V, NH); }

	VecBTC Forward(const VecBTC &inputs);

	void Backward();
};
