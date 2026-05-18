#include "global.h"

#include <cassert>

void operator+=(VecBTC &x, const VecBTC &y) {
	size_t batchs = x.size();
	for (int b = 0; b < batchs; ++b) {
		assert(x[b].rows() == x[b].rows() && x[b].cols() == y[b].cols());
		x[b] += y[b];
	}
}
