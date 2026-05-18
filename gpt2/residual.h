#pragma once

#include "global.h"


class Residual {
public:
	using InputsGrad = std::pair<VecBTC,VecBTC>;
	Residual(){}
	Residual(size_t B,size_t T,size_t C){}
	VecBTC Forward(const VecBTC &, const VecBTC &);
	InputsGrad Backward(const VecBTC& d_outputs);

};
