#pragma once
#include "global.h"

class MatMul {
public:
	MatMul() {}
	MatMul(size_t C, size_t OC) : weight(C,OC), bias(OC) {}
	FORWARD_NO_DISCARD
	VecBtc forward(const VecBtc &);
	void backward();

public:
	Matf weight;
	Vecf bias;
};
