#pragma once
#include "global.h"
#include "matmul.h"

#include <cmath>
#include <vector>

struct AdamWParam {
	float lr;
	float beta1;
	float beta2;
	float eps;
	float weight_decay;
	int t;
};

template <class param_datas_type, class param_grads_type>
void AdamW(param_datas_type &data, param_grads_type &grad_data, param_grads_type &m, param_grads_type &v, size_t size, const AdamWParam &aw_params) {
	for (size_t i = 0; i < size; ++i) {
		float param = data[i];
		float grad = grad_data[i];

		float m_ = aw_params.beta1 * m[i] + (1.0f - aw_params.beta1) * grad;
		float v_ = aw_params.beta2 * v[i] + (1.0f - aw_params.beta2) * grad * grad;

		float m_hat = m_ / (1.0f - std::powf(aw_params.beta1, aw_params.t));
		float v_hat = v_ / (1.0f - std::powf(aw_params.beta2, aw_params.t));

		m[i] = m_;
		v[i] = v_;

		data[i] -= aw_params.lr * (m_hat / (std::sqrtf(v_hat) + aw_params.eps) + aw_params.weight_decay * param);
	}
}

