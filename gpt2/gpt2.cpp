#include "gpt2.h"

#include "encoder.h"
#include "fmt/core.h"
#include "global.h"

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

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
void GPT2::Init() {
	fmt::println("{}", fmt::styled("[GPT-2]", fmt::fg(fmt::color::green) | fmt::emphasis::bold));
	config_.Print();
	encoder_ = Encoder{ config_.vocab_size, config_.max_seq_len, config_.channels };
	layernorm_ = LayerNorm{ config_.vocab_size, config_.max_seq_len, config_.channels };
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

	Init();
	//...
}

void GPT2::forward(Mat &inputs, Mat &targets) {
	VecBtc enc = encoder_.forward(inputs);
	fmt::println("encoder forward sharp: ({} ,{} , {})", enc.size(), enc[1].rows(), enc[0].cols());

	fmt::println("enc[0] : \n{}", fmt::streamed(enc[0].block<2, 2>(1, 1)));

	// for (int l = 0; l < config_.num_layers; ++l) {
	//

	VecBtc ln_f = layernorm_.forward(enc);
	fmt::println("layernorm forward sharp: ({} ,{} , {})", ln_f.size(), ln_f[0].rows(), ln_f[0].cols());

	fmt::println("ln_f[0]: \n{}", fmt::streamed(ln_f[0].block<2, 2>(1, 1)));

	// }

	// DEBUG_PRINT_F();
}
