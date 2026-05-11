#include "attention.h"

#include "global.h"
#include "softmax.h"

#include <cassert>
#include <cstddef>

Attention::THc Attention::ScaledDotAttention(int b, int h, const Matf &qq, const Matf &kk, const Matf &vv) {
	size_t seq_len = vv.rows();
	size_t head_channels = vv.cols();
	float scale = 1.f / std::sqrt(head_channels);
	pre_att[b][h] = (qq * kk.transpose()) * scale;

	att[b][h] = softmax(pre_att[b][h]);

	THc res = att[b][h] * vv;
	return res;
}

VecBTC Attention::Forward(const VecBTC &inputs) {
	size_t batchs = inputs.size();
	size_t seq_len = inputs.front().rows();
	size_t channels_3 = inputs.front().cols();
	assert(channels_3 % 3 == 0);
	size_t channels = channels_3 / 3;
	VecBTC output(batchs, Matf(seq_len, channels));
	size_t num_heads = att.front().size();

	assert(channels % num_heads == 0);
	//it's q's dimensions each attention head that is same as k,v;
	size_t q_dim = channels / num_heads;

	for (int b = 0; b < batchs; ++b) {
		//it's Q part
		const Matf &qQ = inputs[b].block(0, 0, seq_len, channels);
		const Matf &kK = inputs[b].block(0, channels, seq_len, channels);
		const Matf &vV = inputs[b].block(0, 2 * channels, seq_len, channels);
		for (int h = 0; h < num_heads; ++h) {
			const Matf &qq = qQ.block(0, h * q_dim, seq_len, q_dim);
			const Matf &kk = kK.block(0, h * q_dim, seq_len, q_dim);
			const Matf &vv = vV.block(0, h * q_dim, seq_len, q_dim);
			output[b].block(0, h * q_dim, seq_len, q_dim) = ScaledDotAttention(b, h, qq, kk, vv);
		}
	}

	return output;
}

void Attention::Backward(){

}
