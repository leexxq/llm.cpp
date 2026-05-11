#pragma once

#include "global.h"

class Residual {
public:
	VecBTC Forward(const VecBTC &, const VecBTC &);
	void Bacward();
};
