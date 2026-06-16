#pragma once
#include "global.h"

FORWARD_NO_DISCARD
VecBTC MatMulForward(const VecBTC &inputs, const Matf &weight);
FORWARD_NO_DISCARD
VecBTC MatMulForward(const VecBTC &inputs, const Matf &weight, const Vecf &bias);

void MatMulBackward(const VecBTC &d_outputs, const VecBTC &inputs, const Matf &weight,
		VecBTC &d_inputs, Matf &d_weight, bool trans, size_t B, size_t T, size_t C, size_t Oc);

class MatMul {
public:
	// d_inputs d_weight d_bias
	using InputsGrad = VecBTC;
	MatMul() {}
	MatMul(size_t C, size_t OC) : weight(Matf::Zero(C, OC)), bias(Vecf::Zero(OC)), d_weight(Matf::Zero(C, OC)), d_bias(Vecf::Zero(OC)) {}
	FORWARD_NO_DISCARD
	VecBTC Forward(const VecBTC &);
	BACKWARD_NO_DISCARD
	InputsGrad Backward(const VecBTC &d_output, const VecBTC &inputs);

public:
	Matf weight;
	Vecf bias;

	Matf d_weight;
	Vecf d_bias;
};
