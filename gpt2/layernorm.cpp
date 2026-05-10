#include "layernorm.h"

#include "global.h"


VecBtc LayerNorm::forward(const VecBtc &inputs) {
	size_t batchs = inputs.size();
	size_t seq_len = inputs.front().rows();
	size_t channels = inputs.front().cols();

	VecBtc output(batchs, Mat(seq_len, channels));
	for (int b = 0; b < batchs; ++b) {
		const Mat &tc = inputs[b];
		for (int t = 0; t < seq_len; ++t) {
			float mean_axis = tc.row(t).mean();
			mean[b](t) = mean_axis;
			Vecf shift = tc.row(t).array() - mean_axis;
			float rstd_axis = shift.array().square().sum() / channels;

			rstd_axis = 1.f / std::sqrt(rstd_axis + eps);
			rstd[b](t) = rstd_axis;

			output[b].row(t) = shift.cwiseProduct(gamma) * rstd_axis + beta;
		}
	}

	return output;
}
