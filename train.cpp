#include "gpt2/dataloader.h"
#include "gpt2/gpt2.h"

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <vector>

int main() {
	namespace fs = std::filesystem;
	fs::path tiny_stories_train{ "data/tinystories/TinyStories_train.bin" };
	fs::path tiny_stories_val{ "data/tinystories/TinyStories_val.bin" };
	fs::path tiny_shakespeare_train{ "data/tinyshakespeare/tiny_shakespeare_train.bin" };
	fs::path tiny_shakespeare_val{ "data/tinyshakespeare/tiny_shakespeare_val.bin" };
	fs::path train_tokens = exists(tiny_shakespeare_train) ? tiny_shakespeare_train : tiny_stories_train;
	fs::path val_tokens = exists(tiny_shakespeare_val) ? tiny_shakespeare_val : tiny_stories_val;
	fs::path checkpoint_path{ "data/gpt2_124M.bin" };
	GPT2 gpt2{ checkpoint_path };
	size_t B = 4;
	size_t T = 64;
	DataLoader train_loader{ train_tokens, B, T, 0, 1, true };
	DataLoader val_loader{ val_tokens, B, T, 0, 1, false };
	std::cout << "train dataset num_batches: " << train_loader.num_tokens / (B * T) << std::endl;
	std::cout << "val dataset num_batches: " << val_loader.num_tokens / (B * T) << std::endl;
	int val_num_batches = 5;

	uint64_t rng_state = 1337;
	std::vector<int> gen_tokens(B * T);
	constexpr int genT = 64;

	// for (int step = 0; step < 40; ++step) {
	train_loader.NextBatch();
	gpt2.forward(train_loader.inputs, train_loader.targets);
	// }

	return 0;
}
