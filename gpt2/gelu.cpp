#include "gelu.h"

#include "global.h"

VecBTC GELU::Forward(const VecBTC &inputs) {
	size_t batchs = inputs.size();
	size_t seq_len = inputs.front().rows();
	size_t channels = inputs.front().cols();
	VecBTC output(batchs);

	for (int b = 0; b < batchs; ++b) {
		const Matf &x = inputs[b];
		output[b] = 0.5f * x.array() * (1 + (kconstant2 * (x.array() + kconstant1 * x.array().pow(3))).tanh());
	}

	return output;
}

GELU::GradsType GELU::Bacward(const VecBTC &d_ouputs, const VecBTC &inputs, const VecBTC &outputs) {
	size_t batchs = inputs.size();
	size_t seq_len = inputs.front().rows();
	size_t channels = inputs.front().cols();
	VecBTC d_inputs = makeVecBTC(batchs,seq_len,channels);
	for (int b = 0; b < batchs; ++b) {
		const Matf &x = inputs[b];
		auto cube = kconstant2 * x.array().pow(3);
		auto tanh_arg = kconstant1 * (x.array() + cube);
		auto tanhf_out = tanh_arg.tanh();
		auto coshf_out = tanh_arg.cosh();
		auto sech_out = 1.0f / (coshf_out * coshf_out);
		auto local_grad = 0.5f * (1.0f + tanhf_out) + x.array() * 0.5f * sech_out * kconstant1 * (1.0f + 3.0f * kconstant2 * x.array() * x.array());
		d_inputs[b].array() = local_grad.array() * d_ouputs[b].array();
	}
	return d_inputs;
}
