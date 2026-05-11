#pragma once

#include "global.h"

class GELU {
public:
	VecBTC Forward(const VecBTC &inputs);

	void Bacward();

};
