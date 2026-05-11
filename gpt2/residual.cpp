#include "residual.h"

#include "global.h"

VecBTC Residual::Forward(const VecBTC &inputs1, const VecBTC &inputs2) {
	VecBTC output(inputs1);
	size_t batchs = inputs2.size();

	for (int b = 0; b < batchs; ++b) {
		output[b] += inputs2[b];
	}
	return output;
}

void Residual::Bacward() {
}
