#pragma once

#include "global.h"

#include <cstddef>

class Attention {
private:
	using THc = Matf;
	using TT = Matf;
	using VecBHTT = VecBHTC;
	using VecBT3C = VecBTC;
	THc CausalScaledDotAttention(int b, int h, const Matf &qq, const Matf &kk, const Matf &vv);
	void CausalScaledDotAttentionBackward(VecBT3C& d_inputs ,const VecBTC &d_outputs, int b, int h, const Matf &qq, const Matf &kk, const Matf &vv);

public:
	Attention() {}
	Attention(size_t B, size_t T, size_t C, size_t NH) :
			pre_att(makeZero(B, NH, T, T)),
			att(makeZero(B, NH, T, T))
			 {
	}

	VecBTC Forward(const VecBTC &inputs);
	VecBTC Backward(const VecBTC &d_outputs, const VecBTC &inputs);

	VecBHTT pre_att;
	VecBHTT att;
};
