#include "cross_entropy.h"

#include "global.h"

float CrossEntropy::Forward(const VecBTV &probs, const Mati &targets) {
	float mean_loss;
	size_t batchs = targets.rows();
	size_t seq_len = targets.cols();

	for (int b = 0; b < batchs; ++b) {
		for (int t = 0; t < seq_len; ++t) {
			losses[b](t) = -logf(probs[b](t,targets(b, t)));
			mean_loss += losses[b](t);
		}
	}
	return mean_loss / (batchs * seq_len);
}

void CrossEntropy::Backward() {}
