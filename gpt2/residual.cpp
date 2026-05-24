#include "residual.h"

#include "global.h"

VecBTC Residual::Forward(const VecBTC &inputs1, const VecBTC &inputs2) {

	size_t batchs = inputs1.size();
	VecBTC outputs(batchs);
	for (int b = 0; b < batchs; ++b) {
		outputs[b].noalias() = inputs1[b] + inputs2[b];
	}

	return outputs;
}

Residual::InputsGrad Residual::Backward(const VecBTC &d_outputs) {
	auto [B, T, C] = GetShape(d_outputs);
	VecBTC d_inputs1 = makeVecBTC(B, T, C);
	VecBTC d_inputs2 = makeVecBTC(B, T, C);
	for (int b = 0; b < B; ++b) {
		d_inputs1[b].noalias() = d_outputs[b];
		d_inputs2[b].noalias() = d_outputs[b];
	}
	return std::make_pair(d_inputs1, d_inputs2);
}
