#include "matmul.h"


#include "global.h"

#include <cassert>
#include <cstddef>

VecBTC MatMulForward(const VecBTC &inputs, const Matf &weight) {
	return MatMulForward(inputs, weight, Vecf());
}



VecBTC MatMulForward(const VecBTC &inputs, const Matf &weight, const Vecf &bias) {
	size_t batchs = inputs.size();
	size_t seq_len = inputs.front().rows();
	size_t channels = inputs.front().cols();
	size_t oc = weight.cols();
	VecBTC outputs(batchs, Matf(seq_len, oc));

	
	for (int b = 0; b < batchs; ++b) {
		const Matf &tc = inputs[b];
		outputs[b].noalias() = tc * weight;
	}

	if (bias.size() != 0) {
		assert(bias.size() == oc);
		for (int b = 0; b < batchs; ++b) {
			for (int t = 0; t < seq_len; ++t) {
				outputs[b].row(t) += bias.transpose();
			}
		}
	}
	return outputs;
}


VecBTC MatMulBackward(const VecBTC &d_outputs, const VecBTC &inputs, const Matf &weight, const Vecf &bias, Matf &d_weight, Vecf &d_bias) {
	auto [B, T, C] = GetShape(inputs);
	size_t Oc = weight.cols();

	assert(d_outputs.front().rows() == T);
	assert(d_outputs.front().cols() == Oc);

	VecBTC d_inputs = makeZero(B, T, C);

	MatMulBackward(d_outputs, inputs, weight, d_inputs, d_weight, false, B, T, C, Oc);

	for (int b = 0; b < B; ++b) {
		for (int i = 0; i < Oc; ++i) {
			d_bias[i] += d_outputs[b].col(i).sum();
		}
	}

	return d_inputs;
}

void MatMulBackward(const VecBTC &d_outputs, const VecBTC &inputs, const Matf &weight,
		VecBTC &d_inputs, Matf &d_weight, bool trans, size_t B, size_t T, size_t C, size_t Oc) {
	for (int b = 0; b < B; ++b) {
		if (trans) {
			d_weight.transpose().block(0, 0, C, Oc).noalias() += inputs[b].block(0, 0, T, C).transpose() * d_outputs[b].block(0, 0, T, Oc); //(C,T) * (T,Oc) = (C,Oc)
			d_inputs[b].block(0, 0, T, C).noalias() += d_outputs[b].block(0, 0, T, Oc) * weight.transpose().block(0, 0, C, Oc).transpose(); // (T,Oc) * (Oc,C) = (T,C)
		} else {
			d_weight.block(0, 0, C, Oc).noalias() += inputs[b].block(0, 0, T, C).transpose() * d_outputs[b].block(0, 0, T, Oc); //(C,T) * (T,Oc) = (C,Oc)
			d_inputs[b].block(0, 0, T, C).noalias() += d_outputs[b].block(0, 0, T, Oc) * weight.block(0, 0, C, Oc).transpose(); // (T,Oc) * (Oc,C) = (T,C)
		}
	}
}

VecBTC MatMul::Forward(const VecBTC &inputs) {
	return MatMulForward(inputs, weight, bias);
}

MatMul::InputsGrad MatMul::Backward(const VecBTC &d_outputs, const VecBTC &inputs) {
	return MatMulBackward(d_outputs, inputs, weight, bias, d_weight, d_bias);
}
