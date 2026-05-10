#include "matmul.h"

#include "global.h"

VecBtc MatMul::forward(const VecBtc &inputs) {
	size_t batchs = inputs.size();
	size_t seq_len = inputs.front().rows();
	size_t channels = inputs.front().cols();
	size_t oc = weight.cols();
	VecBtc output(batchs, Matf(seq_len, oc));
	for (int b = 0; b < batchs; ++b) {
		const Matf &tc = inputs[b];
		output[b] = tc * weight;
		for (int t = 0; t < seq_len; ++t) {
			output[b].row(t) = output[b].row(t) + bias.transpose();
		}
	}

	return output;
}

void MatMul::backward() {}
