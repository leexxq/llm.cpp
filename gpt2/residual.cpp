#include "residual.h"

#include "global.h"

VecBTC Residual::Forward(const VecBTC &inputs1, const VecBTC &inputs2) {
	VecBTC output(inputs1);
	size_t batchs = inputs2.size();

	for (int b = 0; b < batchs; ++b) {
		output[b] += inputs2[b];
	}
	return output;
}

Residual::InputsGrad Residual::Backward(const VecBTC &d_outputs) {
	auto [B, T, C] = GetShape(d_outputs);
	VecBTC d_inputs1 = makeVecBTC(B, T, C);
	VecBTC d_inputs2 = makeVecBTC(B, T, C);
	for (int b = 0; b < B; ++b) {
		d_inputs1[b] = d_outputs[b];
		d_inputs2[b] = d_outputs[b];
	}
	return std::make_pair(d_inputs1, d_inputs2);
}
