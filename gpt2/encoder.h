#pragma once
#include "global.h"

#include <cstddef>

class Encoder {
public:
	using MatTC = Matf;
	using MatVC = Matf;
	MatVC wte;
	MatTC wpe;
	Encoder() {}
	Encoder(size_t Vp, size_t maxT, size_t C) : wte(Matf::Zero(Vp, C)),
												wpe(Matf::Zero(maxT, C)),
												d_wte(Matf::Zero(Vp, C)),
												d_wpe(Matf::Zero(maxT, C)) {}
	FORWARD_NO_DISCARD
	VecBTC Forward(const Mati &input) const;
	void Backward(const VecBTC &d_output, const Mati &input);

	MatVC d_wte;
	MatTC d_wpe;
};
