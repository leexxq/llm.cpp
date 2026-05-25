#include "cross_entropy.h"
#include "global.h"

#include <cmath>

void CrossEntropySoftmaxBackward(VecBTC &d_logits, const VecBTC &probs, const Mati &targets) {
	size_t B = probs.size();
	size_t T = probs.front().rows();
	size_t V = probs.front().cols();
	for (int b = 0; b < B; ++b) {
		for (int t = 0; t < T; ++t) {
			for (int c = 0; c < V; ++c) {
				d_logits[b](t, c) += probs[b](t, c) - (targets(b, t) == c ? 1.0f : 0.0f);
			}
		}
		d_logits[b] /= B * T;
	}
}

float CrossEntropy::Forward(const VecBTV &probs, const Mati &targets) {
	float mean_loss;
	size_t batchs = targets.rows();
	size_t seq_len = targets.cols();

	for (int b = 0; b < batchs; ++b) {
		for (int t = 0; t < seq_len; ++t) {
			losses[b](t) = -std::log(probs[b](t, targets(b, t)));
			mean_loss += losses[b](t);
		}
	}
	return mean_loss / (batchs * seq_len);
}

void CrossEntropy::Backward() {}
