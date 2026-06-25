#include "tokenizer.h"
#include "log.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

Tokenizer::Tokenizer(const std::filesystem::path &checkpoint_path) {
	std::ifstream f{ checkpoint_path, std::ios::binary };
	if (!f.is_open()) {
		ERROR_PRINTLN("---");
		ERROR_PRINTLN("WARNING: Failed to open the tokenizer file {}", fmt::streamed(checkpoint_path));
		ERROR_PRINTLN("The Tokenizer is a new feature added April 14 2024.");
		ERROR_PRINTLN("Re-run `python train_gpt2.py` to write it");
		ERROR_PRINTLN("---");
		init_ok_ = false;
		return;
	}
	uint32_t header[256];
	f.read(reinterpret_cast<char *>(header), sizeof(uint32_t) * 256);
	assert(header[0] == 20240328);
	int version = header[1];
	size_t vocab_size = header[2];
	if (version == 1) {
		assert(vocab_size == 50257);
		eot_token = 50256;
	} else if (version == 2) {
		eot_token = header[3];
	} else {
		ERROR_PRINTLN("Tokenizer model file {} has bad version: {}", fmt::streamed(checkpoint_path), version);
		exit(EXIT_FAILURE);
	}

	unsigned char length;
	token_table_.reserve(vocab_size);
	for (uint32_t i = 0; i < vocab_size; ++i) {
		f.read(reinterpret_cast<char *>(&length), sizeof(unsigned char));
		assert(length > 0);
		std::string token_bytes;
		token_bytes.resize(length);
		f.read(token_bytes.data(), sizeof(char) * length);
		token_table_.emplace_back(token_bytes);
	}
	init_ok_ = true;
}

std::vector<int> Tokenizer::encode(const std::string &str) {
	return std::vector<int>(str.size());
}
std::string Tokenizer::decode(const std::vector<int> &codes) {
	std::string res;
	for (auto &code : codes) {
		if (code < token_table_.size()) {
			res.append(token_table_[code]);
		} else {
			return std::string();
		}
	}
	return res;
}

std::string Tokenizer::decode(int code) {
	std::vector<int> codes(1, code);
	return Tokenizer::decode(codes);
}
