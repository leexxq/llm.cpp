#pragma once


struct AdamWParams {
	float lr;
	float beta1;
	float beta2;
	float eps;
	float weight_decay;
	int t;
};


void AdamW(float* data, float * grad_data, float* m, float* v, size_t size, const AdamWParams &aw_params) ;