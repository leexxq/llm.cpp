#pragma once
#include "global.h"

#include <cstddef>

class Encoder {
public:
	Matf wte;
	Matf wpe;
	Encoder() {}
	Encoder(size_t Vp, size_t maxT, size_t C) {
		wte = Matf(Vp, C);
		wpe = Matf(maxT, C);
	}

	FORWARD_NO_DISCARD
	VecBTC Forward(const Matf &input) const;
	void Backward();
};
