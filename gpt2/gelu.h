#pragma once

#include "global.h"

class GELU {

	constexpr static float kconstant2 = 0.044715f;
	constexpr static float kconstant1 = 0.7978845608f; //sqrt(2/pi)
public:
	//d_inputs
	using GradsType = VecBTC;
	FORWARD_NO_DISCARD
	VecBTC Forward(const VecBTC &inputs);

	BACKWARD_NO_DISCARD
	VecBTC Bacward(const VecBTC &d_ouputs,const VecBTC& inputs ,const VecBTC& output);

};
