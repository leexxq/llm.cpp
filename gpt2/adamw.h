#pragma once

#include <cstddef>
struct AdamWParams {
	float lr;
	float beta1;
	float beta2;
	float eps;
	float weight_decay;
	int t;
};


void AdamW(float* data, float const* grad_data, float* m, float* v, size_t size, const AdamWParams &aw_params) ;