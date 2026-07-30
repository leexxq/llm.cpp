#include "attention.h"
#include <cassert>
#include <cstddef>

Attention::THc Attention::CausalScaledDotAttention(int b, int h, const Matf &qq, const Matf &kk, const Matf &vv) {
	size_t seq_len = vv.rows();
	size_t head_channels = vv.cols();
	float scale = 1.f / std::sqrt(head_channels);
	Matf output = Matf(seq_len, head_channels);

	// before
	// for (int t = 0; t < seq_len; ++t) {
	// 	float maxval = -100000.0f;
	// 	for (int t1 = 0; t1 <= t; ++t1) {
	// 		float val = qq.row(t).dot(kk.row(t1)) * scale;
	// 		pre_att[b][h](t, t1) = val;
	// 		maxval = std::max(maxval, val);
	// 	}
	//
	// 	Vecf exp = (pre_att[b][h].row(t).segment(0, t + 1).array() - maxval).exp();
	// 	float expsum_inv = exp.sum();
	// 	expsum_inv = expsum_inv == 0.0f ? 0.0f : 1.0f / expsum_inv;
	// 	att[b][h].row(t).segment(0, t + 1) = exp.segment(0, t + 1) * expsum_inv;
	// 	for (int t1 = t + 1; t1 < seq_len; ++t1) {
	// 		att[b][h](t, t1) = 0.0f;
	// 	}
	//
	// 	for (int t1 = 0; t1 <= t; ++t1) {
	// 		output.row(t) += att[b][h](t, t1) * vv.row(t1);
	// 	}
	// }
	//

	// optim
	pre_att[b][h].triangularView<Eigen::Lower>() = qq * kk.transpose() * scale;

	for (int t = 0; t < seq_len; ++t) {
		auto tmp_pre_att = pre_att[b][h].row(t).segment(0, t + 1);
		auto tmp_att = att[b][h].row(t).segment(0, t + 1);
		float maxval = tmp_pre_att.maxCoeff();
		Vecf exp = (tmp_pre_att.array() - maxval).exp();
		float expsum_inv = exp.sum();
		expsum_inv = expsum_inv == 0.0f ? 0.0f : 1.0f / expsum_inv;
		tmp_att = exp.segment(0, t + 1) * expsum_inv;
		att[b][h].row(t).segment(t + 1, seq_len - t - 1).setZero();
	}
	output = att[b][h].triangularView<Eigen::Lower>() * vv;

	return output;
}



void Attention::CausalScaledDotAttentionBackward(VecBT3C &d_inputs, const VecBTC &d_outputs, int b, int h, const Matf &qq, const Matf &kk, const Matf &vv) {
	size_t seq_len = vv.rows();
	size_t head_channels = vv.cols();
	float scale = 1.f / std::sqrt(head_channels);
	size_t channels_3 = d_inputs.front().cols();
	assert(channels_3 % 3 == 0);
	size_t channels = channels_3 / 3;

	using Eigen::Ref;

	const Matf &d_output = d_outputs[b].block(0, h * head_channels, seq_len, head_channels);

	Ref<Matf> d_qq = d_inputs[b].block(0, 0, seq_len, channels).block(0, h * head_channels, seq_len, head_channels);
	Ref<Matf> d_kk = d_inputs[b].block(0, channels, seq_len, channels).block(0, h * head_channels, seq_len, head_channels);
	Ref<Matf> d_vv = d_inputs[b].block(0, 2 * channels, seq_len, channels).block(0, h * head_channels, seq_len, head_channels);

	const Matf &att_bh = att[b][h];

	// before

	// Matf d_att = Matf::Zero(seq_len, seq_len);
	// Matf d_pre_att = Matf::Zero(seq_len, seq_len);
	//
	// for (int t = 0; t < seq_len; ++t) {
	// 	for (int t1 = 0; t1 <= t; ++t1) {
	// 		d_vv.row(t1) += att_bh(t, t1) * d_output.row(t);
	// 		d_att(t, t1) += d_output.row(t).dot(vv.row(t1));
	// 	}
	//
	// 	for (int t1 = 0; t1 <= t; ++t1) {
	// 		for (int t2 = 0; t2 <= t; ++t2) {
	// 			float local_derivative = ((t2 == t1 ? 1.0 : 0.0f) - att_bh(t, t2)) * att_bh(t, t1);
	// 			d_pre_att(t, t2) += local_derivative * d_att(t, t1);
	// 		}
	// 	}
	//
	// 	for (int t1 = 0; t1 <= t; ++t1) {
	// 		float factor = scale * d_pre_att(t, t1);
	// 		d_qq.row(t) += factor * kk.row(t1);
	// 		d_kk.row(t1) += factor * qq.row(t);
	// 	}
	//
	// }

	//optim

	Matf d_att = Matf(seq_len, seq_len);

	d_att.triangularView<Eigen::Lower>() = d_output * vv.transpose();

	Matf d_pre_att = Matf::Zero(seq_len, seq_len);

	for (int t = 0; t < seq_len; ++t) {
		for (int t1 = 0; t1 <= t; ++t1) {
			for (int t2 = 0; t2 <= t; ++t2) {
				float local_derivative = ((t2 == t1 ? 1.0 : 0.0f) - att_bh(t, t2)) * att_bh(t, t1);
				d_pre_att(t, t2) += local_derivative * d_att(t, t1);
			}
		}
	}

	d_vv = att_bh.triangularView<Eigen::Lower>().transpose() * d_output;

	d_kk = d_pre_att.triangularView<Eigen::Lower>().transpose() * qq * scale;

	d_qq = d_pre_att.triangularView<Eigen::Lower>() * kk * scale;

	// std::cout << d_att << std::endl;
	// std::cout << "---------" << std::endl;
	// std::cout << d_pre_att << std::endl;
	// std::cout << "---------" << std::endl;
	// std::cout << d_qq << std::endl;
	// std::cout << "---------" << std::endl;
	// std::cout << d_kk << std::endl;
	// std::cout << "---------" << std::endl;
	// std::cout << d_vv << std::endl;
	// std::cout << "---------" << std::endl;
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
	size_t head_channels = channels / num_heads;

	for (int b = 0; b < batchs; ++b) {
		//it's Q part
		const Matf &qQ = inputs[b].block(0, 0, seq_len, channels);
		const Matf &kK = inputs[b].block(0, channels, seq_len, channels);
		const Matf &vV = inputs[b].block(0, 2 * channels, seq_len, channels);
		for (int h = 0; h < num_heads; ++h) {
			const Matf &qq = qQ.block(0, h * head_channels, seq_len, head_channels);
			const Matf &kk = kK.block(0, h * head_channels, seq_len, head_channels);
			const Matf &vv = vV.block(0, h * head_channels, seq_len, head_channels);
			output[b].block(0, h * head_channels, seq_len, head_channels) = CausalScaledDotAttention(b, h, qq, kk, vv);
		}
	}

	return output;
}

VecBTC Attention::Backward(const VecBTC &d_outputs, const VecBTC &inputs) {
	auto [B, T, C3] = GetShape(inputs);
	assert(C3 % 3 == 0);
	size_t C = C3 / 3;
	size_t num_heads = att.front().size();
	size_t head_dim = C / num_heads;
	VecBTC d_inputs = makeZero(B, T, C3);
	for (int b = 0; b < B; ++b) {
		//it's Q part
		const Matf &qQ = inputs[b].block(0, 0, T, C);
		const Matf &kK = inputs[b].block(0, C, T, C);
		const Matf &vV = inputs[b].block(0, 2 * C, T, C);

		for (int h = 0; h < num_heads; ++h) {
			const Matf &qq = qQ.block(0, h * head_dim, T, head_dim);
			const Matf &kk = kK.block(0, h * head_dim, T, head_dim);
			const Matf &vv = vV.block(0, h * head_dim, T, head_dim);
			const Matf &d_output = d_outputs[b].block(0, h * head_dim, T, head_dim);
			CausalScaledDotAttentionBackward(d_inputs, d_outputs, b, h, qq, kk, vv);
		}
	}
	return d_inputs;
}
