#pragma once
#include "global.h"

class LayerNorm {
public:
	LayerNorm() {}
	LayerNorm(size_t B, size_t T, size_t C, float eps = 1e-5f) :
			mean(B, Vecf(T)),
			rstd(B, Vecf(T)),
			gamma(Vecf::Ones(C)),
			beta(Vecf::Zero(C)),
			d_gamma(Vecf::Zero(C)),
			d_beta(Vecf::Zero(C)),
			eps{ eps } {}

	FORWARD_NO_DISCARD VecBTC Forward(const VecBTC &inputs);
	BACKWARD_NO_DISCARD VecBTC Backward(const VecBTC &d_outputs, const VecBTC &inputs);

	float eps;
	VecBT mean;
	VecBT rstd;
	Vecf gamma;
	Vecf beta;

	Vecf d_gamma;
	Vecf d_beta;
};
