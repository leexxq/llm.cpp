#pragma once
#include "global.h"

VecBTC MatMulForward(const VecBTC &inputs, const Matf &weight, const Vecf &bias);
void MatMulBackward(const VecBTC &inputs, Matf &weight, Vecf &bias);

class MatMul {
public:
	MatMul() {}
	MatMul(size_t C, size_t OC) : weight(C, OC), bias(OC) {}
	FORWARD_NO_DISCARD
	VecBTC Forward(const VecBTC &);
	void Backward();

public:
	Matf weight;
	Vecf bias;
};
