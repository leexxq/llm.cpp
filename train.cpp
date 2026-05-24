#include "gpt2/dataloader.h"
#include "gpt2/gpt2.h"

#include <chrono>
#include <cstddef>
#include <cstdlib>
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
	size_t B = 1;
	size_t T = 128;
	GPT2 gpt2{ checkpoint_path, B, T };
	DataLoader train_loader{ train_tokens, B, T, 0, 1, true };
	DataLoader val_loader{ val_tokens, B, T, 0, 1, false };
	std::cout << "train dataset num_batches: " << train_loader.num_tokens / (B * T) << std::endl;
	std::cout << "val dataset num_batches: " << val_loader.num_tokens / (B * T) << std::endl;
	const int val_num_batches = 5;

	uint64_t rng_state = 1337;
	std::vector<int> gen_tokens(B * T);
	constexpr int genT = 64;

	train_loader.NextBatch();
	INFO_PRINTLN("---------------{}---------------", fmt::styled("[Train...]", fmt::fg(fmt::color::green) | fmt::emphasis::bold));
	for (int step = 0; step < 100; ++step) {
		auto start = std::chrono::steady_clock::now();
		auto start_f = std::chrono::steady_clock::now();
		gpt2.Forward(train_loader.inputs, train_loader.targets);
		auto end_f = std::chrono::steady_clock::now();
		gpt2.ZeroGrad();
		auto start_b = std::chrono::steady_clock::now();
		gpt2.Backward();
		auto end_b = std::chrono::steady_clock::now();
		gpt2.Update(1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step + 1);
		auto end = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		auto duration_f = std::chrono::duration_cast<std::chrono::milliseconds>(end_f - start_f);
		auto duration_b = std::chrono::duration_cast<std::chrono::milliseconds>(end_b - start_b);
		INFO_PRINTLN("step {}, mean loss:{}, duration:{}ms, forward duration:{}ms, backward duration:{}ms", step + 1, gpt2.mean_loss, duration.count(), duration_f.count(), duration_b.count());
	}

	return 0;
}
