#include "global.h"

void CrossEntropySoftmaxBackward(VecBTC &d_logits, const VecBTC &probs, const Mati &targets); 

class CrossEntropy {
public:
	using VecBTV = VecBTC;
	CrossEntropy() {}
	CrossEntropy(size_t B, size_t T) : losses(B, Vecf::Zero(T)) {}

	float Forward(const VecBTV &probs, const Mati &targets);
	void Backward();

	VecBT losses;
};
