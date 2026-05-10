#pragma once
#include "global.h"

class LayerNorm {
public:
	LayerNorm() {}
	LayerNorm(size_t B, size_t T, size_t C, float eps = 1e-5f) :
			mean(B, Vecf(T)),
			rstd(B, Vecf(T)),
			gamma(Vecf::Constant(C, 1)),
			beta(Vecf::Constant(C, 0)),
			eps{ eps } {}

	FORWARD_NO_DISCARD VecBtc forward(const VecBtc &inputs);
	void backward();

	float eps;
	VecBt mean;
	VecBt rstd;
	Vecf gamma;
	Vecf beta;
};
