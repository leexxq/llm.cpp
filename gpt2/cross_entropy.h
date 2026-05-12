#include "global.h"

class CrossEntropy {
public:
	using VecBTV = VecBTC;
	CrossEntropy() {}
	CrossEntropy(size_t B, size_t T) : losses(B, Vecf(T)) {}

	float Forward(const VecBTV &probs, const Mati &targets);
	void Backward();

	VecBT losses;
};
