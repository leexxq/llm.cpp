#include "adamw.h"

#include <cmath>
void AdamW(float* data, float * grad_data, float* m, float* v, size_t size, const AdamWParams &aw_params) {
	for (size_t i = 0; i < size; ++i) {
		float param = data[i];
		float grad = grad_data[i];

		float m_ = aw_params.beta1 * m[i] + (1.0f - aw_params.beta1) * grad;
		float v_ = aw_params.beta2 * v[i] + (1.0f - aw_params.beta2) * grad * grad;

		float m_hat = m_ / (1.0f - std::pow(aw_params.beta1, aw_params.t));
		float v_hat = v_ / (1.0f - std::pow(aw_params.beta2, aw_params.t));

		m[i] = m_;
		v[i] = v_;

		data[i] -= aw_params.lr * (m_hat / (std::sqrt(v_hat) + aw_params.eps) + aw_params.weight_decay * param);
	}
}

