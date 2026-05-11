#pragma once
#include "global.h"

#include <cstddef>

class Encoder {
public:
	Matf wte;
	Matf wpe;
	Encoder() {}
	Encoder(size_t vp, size_t seq_len, size_t c) {
		wte = Matf(vp, c);
		wpe = Matf(seq_len, c);
	}

	FORWARD_NO_DISCARD
	VecBTC Forward(const Matf &input) const;
	void Backward();
};
