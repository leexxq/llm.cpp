#pragma once

#include "cross_entropy.h"
#include "encoder.h"
#include "global.h"
#include "layer.h"
#include "layernorm.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cstddef>
#include <filesystem>
#include <vector>

struct GPT2Config {
public:
	size_t max_seq_len; // max sequence length, e.g. 1024
	size_t vocab_size; // vocab size, e.g. 50257
	size_t padded_vocab_size; // padded to e.g. %128==0, 50304
	size_t num_layers; // number of layers, e.g. 12
	size_t num_heads; // number of heads in attention, e.g. 12
	size_t channels; // number of channels, e.g. 768
					 //
	void Print() const;
};

class GPT2 {
private:
	GPT2Config config_;
	// the weights (parameters) of the model, and their sizes
	Eigen::VectorXf params_memory;
	size_t num_parameters;
	// gradients of the weights
	Eigen::VectorXf grads_memory;
	// buffers for the AdamW optimizer
	Eigen::VectorXf m_memory;
	Eigen::VectorXf v_memory;
	// the activations of the model, and their sizes
	Eigen::VectorXd acts_memory;
	size_t num_activations;
	// gradients of the activations
	Eigen::VectorXd grads_acts_memory;
	// other run state configuration
	// size_t batch_size; // the batch size (B) of current forward pass
	// size_t seq_len; // the sequence length (T) of current forward pass
	Matf inputs_; // the input tokens for the current forward pass
	Mati targets_; // the target tokens for the current forward pass
	float mean_loss; // after a forward pass with targets, will be populated with the mean loss
					 //
	Encoder encoder_;

	LayerNorm layernormf_;

	CrossEntropy loss_;

	std::vector<Layer> layers_;

	size_t B_;
	size_t T_;

	VecBTC encoded_;
	VecLBTC residual_;
	VecBTC lnf_;
	VecBTVp logits_;
	VecBTVp probs_;

	VecBTVp d_logits_;
	VecBTC d_lnf_;
	VecLBTC d_residual3_;
	VecBTC d_encoded;

	std::filesystem::path checkpoint_path_;

private:
	bool _init = false;
	void Init(size_t B, size_t T);

public:
	GPT2() : config_{ 1024, 50257, 50304, 12, 12, 768 } {}
	GPT2(GPT2Config config,size_t B,size_t T) : config_{ config },B_(B),T_(T) {}
	GPT2(const std::filesystem::path &path,size_t B,size_t T);
	GPT2(const GPT2 &gpt2) = delete;
	GPT2(const GPT2 &&gpt2) = delete;
	void Forward(Mati &, Mati &);
	void Backward();
	void ZeroGrad();
	void Update(float lr , float beta1,float beta2,float eps , float weight,int t);
};
