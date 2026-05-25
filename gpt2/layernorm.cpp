#include "layernorm.h"

#include "global.h"

#include <cassert>

VecBTC LayerNorm::Forward(const VecBTC &inputs) {
	size_t batchs = inputs.size();
	size_t seq_len = inputs.front().rows();
	size_t channels = inputs.front().cols();

	VecBTC output(batchs, Matf(seq_len, channels));
	for (int b = 0; b < batchs; ++b) {
		const Matf &tc = inputs[b];
		for (int t = 0; t < seq_len; ++t) {
			float mean_axis = tc.row(t).mean();
			mean[b](t) = mean_axis;
			Vecf shift = tc.row(t).array() - mean_axis;
			float rstd_axis = shift.cwiseSquare().sum();

			rstd_axis /= channels;

			rstd_axis = 1.f / std::sqrt(rstd_axis + eps);

			rstd[b](t) = rstd_axis;

			output[b].row(t) = shift.cwiseProduct(gamma) * rstd_axis + beta;
		}
	}

	return output;
}

VecBTC LayerNorm::Backward(const VecBTC &d_outputs, const VecBTC &inputs) {
	size_t batchs = inputs.size();
	size_t seq_len = inputs.front().rows();
	size_t channels = inputs.front().cols();
	VecBTC d_inputs = makeVecBTC(batchs, seq_len, channels);
	for (int b = 0; b < batchs; ++b) {
		Matf x_norm(seq_len, channels);
		for (int t = 0; t < seq_len; ++t) {
			const float rstd_bt = rstd[b](t);
			const float mean_bt = mean[b](t);
			x_norm.row(t) = (inputs[b].row(t).array() - mean_bt) * rstd_bt;
		}

		for (int i = 0; i < channels; ++i) {
			d_gamma[i] += d_outputs[b].col(i).dot(x_norm.col(i));
			d_beta[i] += d_outputs[b].col(i).sum();
		}

		for (int t = 0; t < seq_len; ++t) {
			const float rstd_bt = rstd[b](t);
			// before
			//
			// for (int c = 0; c < channels; ++c) {
			// 	const float term1 = gamma[c] * d_outputs[b](t, c);
			// 	const float term2 = d_outputs[b].row(t).dot(gamma) / channels;
			// 	const float term3_item2 = gamma.dot(d_outputs[b].row(t).cwiseProduct(x_norm.row(t)));
			// 	const float term3 = (x_norm(t, c) * term3_item2) / channels;
			// 	d_inputs[b](t, c) += rstd_bt * (term1 - term2 - term3);
			// }
			
			//optim

			Vecf gamma_dy = d_outputs[b].row(t).array() * gamma.transpose().array();
			const float gamma_dot_dy = gamma_dy.sum() / channels;
			Vecf term3 = (x_norm.row(t).array() * gamma_dy.dot(x_norm.row(t))) / channels;
			d_inputs[b].row(t) = rstd_bt * ((gamma_dy - term3).array() - gamma_dot_dy);
		}
	}
	return d_inputs;
}
