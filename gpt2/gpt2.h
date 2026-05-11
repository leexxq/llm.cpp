#pragma once

#include "attention.h"
#include "encoder.h"
#include "global.h"
#include "layernorm.h"
#include "matmul.h"

#include <Eigen/Dense>
#include <cstddef>
#include <filesystem>

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

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
struct ParameterTensors {
public:
	Eigen::MatrixXf wte; // (V, C)
	Eigen::MatrixXf wpe; // (maxT, C)
	Eigen::MatrixXf ln1w; // (L, C)
	Eigen::MatrixXf ln1b; // (L, C)
	Eigen::MatrixXf qkvw; // (L, 3*C, C)
	Eigen::MatrixXf qkvb; // (L, 3*C)
	Eigen::MatrixXf attprojw; // (L, C, C)
	Eigen::MatrixXf attprojb; // (L, C)
	Eigen::MatrixXf ln2w; // (L, C)
	Eigen::MatrixXf ln2b; // (L, C)
	Eigen::MatrixXf fcw; // (L, 4*C, C)
	Eigen::MatrixXf fcb; // (L, 4*C)
	Eigen::MatrixXf fcprojw; // (L, C, 4*C)
	Eigen::MatrixXf fcprojb; // (L, C)
	Eigen::MatrixXf lnfw; // (C)
	Eigen::MatrixXf lnfb; // (C)
};

#define NUM_ACTIVATION_TENSORS 23
struct ActivationTensors {
public:
	Eigen::MatrixXf encoded; // (B, T, C)
	Eigen::MatrixXf ln1; // (L, B, T, C)
	Eigen::MatrixXf ln1_mean; // (L, B, T)
	Eigen::MatrixXf ln1_rstd; // (L, B, T)
	Eigen::MatrixXf qkv; // (L, B, T, 3*C)
	Eigen::MatrixXf atty; // (L, B, T, C)
	Eigen::MatrixXf preatt; // (L, B, NH, T, T)
	Eigen::MatrixXf att; // (L, B, NH, T, T)
	Eigen::MatrixXf attproj; // (L, B, T, C)
	Eigen::MatrixXf residual2; // (L, B, T, C)
	Eigen::MatrixXf ln2; // (L, B, T, C)
	Eigen::MatrixXf ln2_mean; // (L, B, T)
	Eigen::MatrixXf ln2_rstd; // (L, B, T)
	Eigen::MatrixXf fch; // (L, B, T, 4*C)
	Eigen::MatrixXf fch_gelu; // (L, B, T, 4*C)
	Eigen::MatrixXf fcproj; // (L, B, T, C)
	Eigen::MatrixXf residual3; // (L, B, T, C)
	Eigen::MatrixXf lnf; // (B, T, C)
	Eigen::MatrixXf lnf_mean; // (B, T)
	Eigen::MatrixXf lnf_rstd; // (B, T)
	Eigen::MatrixXf logits; // (B, T, V)
	Eigen::MatrixXf probs; // (B, T, V)
	Eigen::MatrixXf losses; // (B, T)
};

class GPT2 {
private:
	GPT2Config config_;
	// the weights (parameters) of the model, and their sizes
	ParameterTensors params_;
	size_t param_sizes[NUM_PARAMETER_TENSORS];
	Eigen::VectorXf params_memory;
	size_t num_parameters;
	// gradients of the weights
	ParameterTensors grads;
	Eigen::VectorXf grads_memory;
	// buffers for the AdamW optimizer
	Eigen::VectorXf m_memory;
	Eigen::VectorXf v_memory;
	// the activations of the model, and their sizes
	ActivationTensors acts;
	size_t act_sizes[NUM_ACTIVATION_TENSORS];
	Eigen::VectorXd acts_memory;
	size_t num_activations;
	// gradients of the activations
	ActivationTensors grads_acts;
	Eigen::VectorXd grads_acts_memory;
	// other run state configuration
	size_t batch_size; // the batch size (B) of current forward pass
	size_t seq_len; // the sequence length (T) of current forward pass
	Matf inputs; // the input tokens for the current forward pass
	Matf targets; // the target tokens for the current forward pass
	float mean_loss; // after a forward pass with targets, will be populated with the mean loss
					 //
	Encoder encoder_;
	LayerNorm layernorm_;
	MatMul matmul_;
	Attention attention_;

private:
	void Init(size_t B,size_t T);

public:
	GPT2() : config_{ 1024, 50257, 50304, 12, 12, 768 } {  }
	GPT2(GPT2Config config) : config_{ config } { }
	GPT2(const std::filesystem::path &path);
	GPT2(const GPT2 &gpt2) = delete;
	GPT2(const GPT2 &&gpt2) = delete;
	void Forward(Matf &, Matf &);
};
