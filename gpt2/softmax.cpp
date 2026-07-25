#include "softmax.h"
#include "global.h"


void Softmax(const Matf &mat,Matf& output) {
	size_t cols = mat.cols();
	size_t rows = mat.rows();
	for (int r = 0; r < rows; ++r) {
		float maxval = mat.row(r).maxCoeff();
		output.row(r) = (mat.row(r).array() - maxval).exp();
		float expsum = output.row(r).sum();
		float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
		output.row(r) *= expsum_inv;
	}
}


Matf Softmax(const Matf &mat) {
	size_t cols = mat.cols();
	size_t rows = mat.rows();
	Matf res(rows, cols);
	for (int r = 0; r < rows; ++r) {
		float maxval = mat.row(r).maxCoeff();
		res.row(r) = (mat.row(r).array() - maxval).exp();
		float expsum = res.row(r).sum();
		float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
		res.row(r) *= expsum_inv;
	}
	return res;
}

VecBTC Softmax(const VecBTC &vmat) {
	size_t batch = vmat.size();
	VecBTC res(batch);
	for (int b = 0; b < batch; ++b) {
		res[b] = Softmax(vmat[b]);
	}
	return res;
}
