#include "gpt2.h"

#include "encoder.h"
#include "global.h"
#include "layernorm.h"
#include "matmul.h"
#include "softmax.h"

#include <Eigen/Core>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
void GPT2Config::Print() const {
	INFO_PRINTLN("---------------------------------");
	INFO_PRINTLN("{}: {}", fmt::styled("max_seq_len", fmt::fg(fmt::color::blue)), max_seq_len);
	INFO_PRINTLN("{}: {}", fmt::styled("vocab_size", fmt::fg(fmt::color::blue)), vocab_size);
	INFO_PRINTLN("{}: {}", fmt::styled("padded_vocab_size", fmt::fg(fmt::color::blue)), vocab_size);
	INFO_PRINTLN("{}: {}", fmt::styled("num_layers", fmt::fg(fmt::color::blue)), num_layers);
	INFO_PRINTLN("{}: {}", fmt::styled("num_heads", fmt::fg(fmt::color::blue)), num_heads);
	INFO_PRINTLN("{}: {}", fmt::styled("channels", fmt::fg(fmt::color::blue)), channels);
	INFO_PRINTLN("---------------------------------");
}
void GPT2::Init(size_t B, size_t T) {
	INFO_PRINTLN("{}", fmt::styled("[GPT-2]", fmt::fg(fmt::color::green) | fmt::emphasis::bold));
	config_.Print();
	encoder_ = Encoder{ config_.vocab_size, config_.max_seq_len, config_.channels };
	for (int l = 0; l < config_.num_layers; ++l) {
		layers_.emplace_back(B, T, config_.channels, config_.vocab_size, config_.num_heads);
	}

	layernormf_ = LayerNorm{ B, T, config_.channels };
	loss_ = CrossEntropy{ B, T };
}

GPT2::GPT2(const std::filesystem::path &path) {
	std::ifstream f{ path, std::ios::binary };
	if (!f.is_open()) {
		ERROR_PRINTLN("Error: connot read file: {}", fmt::streamed(path));
		exit(EXIT_FAILURE);
	}

	int model_header[256];

	f.read(reinterpret_cast<char *>(model_header), sizeof(model_header));

	if (model_header[0] != 20240326) {
		ERROR_PRINTLN("Bad magic model file: {}", fmt::streamed(path));
		exit(EXIT_FAILURE);
	}
	if (model_header[1] != 3) {
		ERROR_PRINTLN("Bad version in model file: {}", fmt::streamed(path));
		exit(EXIT_FAILURE);
	}

	// read in hyperparameters
	size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
	config_.max_seq_len = maxT = model_header[2];
	config_.vocab_size = V = model_header[3];
	config_.num_layers = L = model_header[4];
	config_.num_heads = NH = model_header[5];
	config_.channels = C = model_header[6];
	config_.padded_vocab_size = Vp = model_header[7];

	//...
}

void GPT2::Forward(Mati &inputs, Mati &targets) {
	size_t B = inputs.rows(), T = inputs.cols();

	Init(B, T);

	this->inputs = inputs.cast<float>();

	VecBTC encoded = encoder_.Forward(this->inputs);

	DEBUG_PRINTLN("encoder forward sharp: ({} ,{} , {})", encoded.size(), encoded[1].rows(), encoded[0].cols());
	DEBUG_PRINTLN("enc[0] : \n{}", fmt::streamed(encoded[0].block<2, 2>(1, 1)));

	DEBUG_PRINTLN("--------------{}:{}--------------------", fmt::styled("Layer", fmt::fg(fmt::color::green) | fmt::emphasis::bold), 1);
	VecBTC residual = layers_.front().Forward(encoded);

	for (int l = 1; l < config_.num_layers; ++l) {
		DEBUG_PRINTLN("--------------{}:{}--------------------", fmt::styled("Layer", fmt::fg(fmt::color::green) | fmt::emphasis::bold), l + 1);
		residual = layers_[l].Forward(residual);
	}

	VecBTC lnf = layernormf_.Forward(residual);
	DEBUG_PRINTLN("lnf forward sharp: ({} , {} , {})", lnf.size(), lnf[0].rows(), lnf[0].cols());
	DEBUG_PRINTLN("lnf[0]: \n{}", fmt::streamed(lnf[0].block<2, 2>(1, 1)));

	using VecBTVp = VecBTC;
	VecBTVp logits = MatMulForward(lnf, encoder_.wte.transpose(), Vecf());
	DEBUG_PRINTLN("logits forward sharp: ({} , {} , {})", logits.size(), logits[0].rows(), logits[0].cols());
	DEBUG_PRINTLN("logits[0]: \n{}", fmt::streamed(logits[0].block<2, 2>(1, 1)));

	// Vp is the padded vocab size (for efficiency), V is the "real" vocab size
	// example: Vp is 50304 and V is 50257
	using VecBTV = VecBTC;
	VecBTV probs(B);
	assert(logits.front().cols() >= config_.vocab_size);
	for (int i = 0; i < B; ++i) {
		probs[i] = softmax(logits[i].block(0, 0, T, config_.vocab_size));
	}
	DEBUG_PRINTLN("probs forward sharp: ({} , {} , {})", probs.size(), probs[0].rows(), probs[0].cols());
	DEBUG_PRINTLN("probs[0]: \n{}", fmt::streamed(probs[0].block<2, 2>(1, 1)));

	mean_loss = targets.size() > 0 ? loss_.Forward(probs, targets) : -1.0f;

	INFO_PRINTLN("mean loss : {}", mean_loss);
}
