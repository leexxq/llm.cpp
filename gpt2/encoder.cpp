#include "encoder.h"

#include "global.h"

VecBTC Encoder::Forward(const Mati &input) const {
	size_t batchs = input.rows();
	size_t seq_len = input.cols();
	size_t channels = wte.cols();
	VecBTC outputs = VecBTC(batchs);

	for (int b = 0; b < batchs; ++b) {
		outputs[b] = Matf(seq_len, channels);
		for (int t = 0; t < seq_len; ++t) {
			outputs[b].row(t) = wte.row(input(b, t)) + wpe.row(t);
		}
	}

	return outputs;
}

void Encoder::Backward(const VecBTC &d_outputs, const Mati &input) {
	size_t batchs = input.rows();
	size_t seq_len = input.cols();
	size_t channels = wte.cols();
	//before
	// for (int b = 0; b < batchs; ++b) {
	// 	for (int t = 0; t < seq_len; ++t) {
	// 		d_wte.row(input(b, t)) += d_outputs[b].row(t);
	// 		d_wpe.row(t) += d_outputs[b].row(t);
	// 	}
	// }

	// optim
	for (int b = 0; b < batchs; ++b) {
		d_wpe.block(0, 0, seq_len, channels) += d_outputs[b];
		for (int t = 0; t < seq_len; ++t) {
			d_wte.row(input(b, t)) += d_outputs[b].row(t);
		}
	}
}
