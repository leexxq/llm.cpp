#include "encoder.h"
using Eigen::MatrixXf;

VecBTC Encoder::Forward(const Matf &input) const{
	size_t batchs = input.rows();
	size_t seq_len = input.cols();
	size_t channels = wte.cols();
	VecBTC output = VecBTC(batchs);

	for (int b = 0; b < batchs; ++b) {
		output[b] = Matf(seq_len, channels);
		for (int t = 0; t < seq_len; ++t) {
			output[b].row(t) = wte.row(input(b, t)) + wpe.row(t);
		}
	}

	return output;
}
