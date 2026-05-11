#include "gelu.h"

VecBTC GELU::Forward(const VecBTC &inputs) {
	size_t batchs = inputs.size();
	size_t seq_len = inputs.front().rows();
	size_t channels = inputs.front().cols();
	VecBTC output(batchs);
	constexpr float constant1 = 0.044715f;
	constexpr float constant2 = 0.7978845608f; //sqrt(2/pi)

	for (int b = 0; b < batchs; ++b) {
		const Matf &x = inputs[b];
		output[b] = 0.5f * x.array() * (1+ (constant2 * (x.array() + constant1 * x.array().pow(3))).tanh());
	}

	return output;
}
