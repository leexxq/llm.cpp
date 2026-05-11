#pragma once

#include "global.h"

#include <cstddef>

class Attention {
private:
	using THc = Matf;
	using TT = Matf;
	using VecBHTT = VecBHTC;
	THc ScaledDotAttention(int b, int h ,const Matf &qq, const Matf &kk, const Matf &vv);
public:
	Attention() {}
	Attention(size_t B, size_t T, size_t C, size_t NH) : pre_att(B, VecHTC(NH, Matf(T, T))), att(B, VecHTC(NH, Matf(T, C))) {}

	VecBTC Forward(const VecBTC &inputs);
	void Backward();

	VecBHTT pre_att;
	VecBHTC att;
};
