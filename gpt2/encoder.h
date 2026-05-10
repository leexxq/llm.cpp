#pragma once
#include "global.h"

#include <cstddef>

class Encoder {
public:
	Mat wte;
	Mat wpe;
	Encoder() {}
	Encoder(size_t vp, size_t seq_len, size_t c) {
		wte = Mat(vp, c);
		wpe = Mat(seq_len, c);
	}

	FORWARD_NO_DISCARD
	VecBtc forward(const Mat &input) const;
	void backward();
};
