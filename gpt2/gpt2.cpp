#include "gpt2.h"

#include "attention.h"
#include "encoder.h"
#include "fmt/core.h"
#include "global.h"
#include "layernorm.h"
#include "matmul.h"
#include "residual.h"
#include "softmax.h"

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

#include <Eigen/Core>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
void GPT2Config::Print() const {
	fmt::println("---------------------------------");
	fmt::println("{}: {}", fmt::styled("max_seq_len", fmt::fg(fmt::color::blue)), max_seq_len);
	fmt::println("{}: {}", fmt::styled("vocab_size", fmt::fg(fmt::color::blue)), vocab_size);
	fmt::println("{}: {}", fmt::styled("padded_vocab_size", fmt::fg(fmt::color::blue)), vocab_size);
	fmt::println("{}: {}", fmt::styled("num_layers", fmt::fg(fmt::color::blue)), num_layers);
	fmt::println("{}: {}", fmt::styled("num_heads", fmt::fg(fmt::color::blue)), num_heads);
	fmt::println("{}: {}", fmt::styled("channels", fmt::fg(fmt::color::blue)), channels);
	fmt::println("---------------------------------");
}
void GPT2::Init(size_t B, size_t T) {
	fmt::println("{}", fmt::styled("[GPT-2]", fmt::fg(fmt::color::green) | fmt::emphasis::bold));
	config_.Print();
	encoder_ = Encoder{ config_.vocab_size, config_.max_seq_len, config_.channels };
	layernorm1_ = LayerNorm{ B, T, config_.channels };
	qkv_ = MatMul{ config_.channels, 3 * config_.channels };
	attention_ = Attention{ B, T, config_.channels, config_.num_heads };
	att_proj_ = MatMul{ config_.channels, config_.channels };
	residual1_ = Residual{};
	layernorm2_ = LayerNorm{ config_.vocab_size, T, config_.channels };
	fch_ = MatMul{ config_.channels, 4 * config_.channels };
	fch_gelu_ = GELU{};
	fcproj_ = MatMul{ 4 * config_.channels, config_.channels };
	residual2_ = Residual{};
	layernormf_ = LayerNorm{ B, T, config_.channels };
	mm_logits_ = MatMul{
		config_.channels, config_.padded_vocab_size
	};
}

GPT2::GPT2(const std::filesystem::path &path) {
	std::ifstream f{ path, std::ios::binary };
	if (!f.is_open()) {
		fmt::println(stderr, "Error: connot read file: {}", fmt::streamed(path));
		exit(EXIT_FAILURE);
	}

	int model_header[256];

	f.read(reinterpret_cast<char *>(model_header), sizeof(model_header));

	if (model_header[0] != 20240326) {
		fmt::println(stderr, "Bad magic model file: {}", fmt::streamed(path));
		exit(EXIT_FAILURE);
	}
	if (model_header[1] != 3) {
		fmt::println(stderr, "Bad version in model file: {}", fmt::streamed(path));
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

void GPT2::Forward(Matf &inputs, Matf &targets) {
	size_t B = inputs.rows(), T = inputs.cols();

	Init(B, T);

	VecBTC encoded = encoder_.Forward(inputs);
	fmt::println("encoder forward sharp: ({} ,{} , {})", encoded.size(), encoded[1].rows(), encoded[0].cols());
	fmt::println("enc[0] : \n{}", fmt::streamed(encoded[0].block<2, 2>(1, 1)));

	VecBTC &residual = encoded;
	// for (int l = 0; l < config_.num_layers; ++l) {
	//
	VecBTC l_ln1 = layernorm1_.Forward(encoded);
	fmt::println("layernorm forward sharp: ({} , {} , {})", l_ln1.size(), l_ln1[0].rows(), l_ln1[0].cols());
	fmt::println("l_ln1[0]: \n{}", fmt::streamed(l_ln1[0].block<2, 2>(1, 1)));

	// 3C is GPT2's dim before qkv split
	using VecBT3C = VecBTC;
	VecBT3C l_qkv = qkv_.Forward(encoded);
	fmt::println("qkv_ forward sharp: ({} , {} , {})", l_qkv.size(), l_qkv[0].rows(), l_qkv[0].cols());
	fmt::println("l_qkv[0]: \n{}", fmt::streamed(l_qkv[0].block<2, 2>(1, 1)));

	VecBTC l_atty = attention_.Forward(l_qkv);
	fmt::println("attention forward sharp: ({} , {} , {})", l_atty.size(), l_atty[0].rows(), l_atty[0].cols());
	fmt::println("l_atty[0]: \n{}", fmt::streamed(l_atty[0].block<2, 2>(1, 1)));
	fmt::println("-----att's shape is: ({} , {} , {} , {})", attention_.att.size(), attention_.att.front().size(), attention_.att.front().front().rows(), attention_.att.front().front().cols());

	VecBTC l_attproj = att_proj_.Forward(l_atty);
	fmt::println("att_proj forward sharp: ({} , {} , {})", l_attproj.size(), l_attproj[0].rows(), l_attproj[0].cols());
	fmt::println("l_attproj[0]: \n{}", fmt::streamed(l_attproj[0].block<2, 2>(1, 1)));

	VecBTC l_residual2 = residual1_.Forward(residual, l_attproj);
	fmt::println("l_residual2 forward sharp: ({} , {} , {})", l_residual2.size(), l_residual2[0].rows(), l_residual2[0].cols());
	fmt::println("l_residual2[0]: \n{}", fmt::streamed(l_residual2[0].block<2, 2>(1, 1)));

	VecBTC l_ln2 = layernorm2_.Forward(l_residual2);
	fmt::println("l_ln2 forward sharp: ({} , {} , {})", l_ln2.size(), l_ln2[0].rows(), l_ln2[0].cols());
	fmt::println("l_ln2[0]: \n{}", fmt::streamed(l_ln2[0].block<2, 2>(1, 1)));

	using VecBT4C = VecBTC;
	VecBT4C l_fch = fch_.Forward(l_ln2);
	fmt::println("l_fch forward sharp: ({} , {} , {})", l_fch.size(), l_fch[0].rows(), l_fch[0].cols());
	fmt::println("l_fch[0]: \n{}", fmt::streamed(l_fch[0].block<2, 2>(1, 1)));

	VecBT4C l_fch_gelu = fch_gelu_.Forward(l_fch);
	fmt::println("l_fch_gelu forward sharp: ({} , {} , {})", l_fch_gelu.size(), l_fch_gelu[0].rows(), l_fch_gelu[0].cols());
	fmt::println("l_fch_gelu[0]: \n{}", fmt::streamed(l_fch_gelu[0].block<2, 2>(1, 1)));

	VecBTC l_fcproj = fcproj_.Forward(l_fch_gelu);
	fmt::println("l_fcproj forward sharp: ({} , {} , {})", l_fcproj.size(), l_fcproj[0].rows(), l_fcproj[0].cols());
	fmt::println("l_fcproj[0]: \n{}", fmt::streamed(l_fcproj[0].block<2, 2>(1, 1)));

	VecBTC l_residual3 = residual2_.Forward(l_residual2, l_fcproj);
	fmt::println("l_residual3 forward sharp: ({} , {} , {})", l_residual3.size(), l_residual3[0].rows(), l_residual3[0].cols());
	fmt::println("l_residual3[0]: \n{}", fmt::streamed(l_residual3[0].block<2, 2>(1, 1)));

	// }
	VecBTC lnf = layernormf_.Forward(l_residual3);
	fmt::println("lnf forward sharp: ({} , {} , {})", lnf.size(), lnf[0].rows(), lnf[0].cols());
	fmt::println("lnf[0]: \n{}", fmt::streamed(lnf[0].block<2, 2>(1, 1)));

	using VecBTVp = VecBTC;
	VecBTVp logits = mm_logits_.Forward(lnf);
	fmt::println("logits forward sharp: ({} , {} , {})", logits.size(), logits[0].rows(), logits[0].cols());
	fmt::println("logits[0]: \n{}", fmt::streamed(logits[0].block<2, 2>(1, 1)));

	// Vp is the padded vocab size (for efficiency), V is the "real" vocab size
	// example: Vp is 50304 and V is 50257
	using VecBTV = VecBTC;
	VecBTV probs(B);
	assert(logits.front().cols() >= config_.vocab_size);
	for (int i = 0; i < B; ++i) {
		probs[i] = softmax(logits[i].block(0, 0, T, config_.vocab_size));
	}
	fmt::println("probs forward sharp: ({} , {} , {})", probs.size(), probs[0].rows(), probs[0].cols());
	fmt::println("probs[0]: \n{}", fmt::streamed(probs[0].block<2, 2>(1, 1)));

}
