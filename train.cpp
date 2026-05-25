#include "CLI/CLI.hpp"
#include "gpt2/dataloader.h"
#include "gpt2/global.h"
#include "gpt2/gpt2.h"
#include "gpt2/tokenizer.h"

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>
void SafePrint(const std::string &rep) {
	if (rep.empty()) {
		return;
	}
	if (!(std::isprint(rep[0]) || std::isspace(rep[0]))) {
		return;
	}
	INFO_PRINT("{}", rep);
}

int SampleMult(const Vecf &probs, float coin) {
	float cdf = 0.f;
	for (int i = 0; i < probs.size(); ++i) {
		cdf += probs[i];
		if (coin < cdf) {
			return i;
		}
	}
	return probs.size() - 1;
}


int main(int argc, char **argv) {

	CLI::App app{"Train parameters"};

	std::string datasets_dir;
	app.add_option("-d,--datasets",datasets_dir,"datasets directory");

	size_t B = 4;
	size_t T = 64;
	size_t genT = 64;
	size_t val_num_batches = 5;
	size_t iterations = 40;
	app.add_option("-b,--batch",B,"training batchs");
	app.add_option("-t,--seq_length",T,"train tokens length");
	app.add_option("--gen_token",genT,"generative text's tokens length");
	app.add_option("--val_sets_batch",val_num_batches,"validate sets's batch for test");
	app.add_option("-i,--iteration",iterations,"train iterations number");

	CLI11_PARSE(app,argc,argv);

	namespace fs = std::filesystem;

	fs::path tiny_stories_train{ "data/tinystories/TinyStories_train.bin" };
	fs::path tiny_stories_val{ "data/tinystories/TinyStories_val.bin" };
	fs::path tiny_shakespeare_train{ "data/tinyshakespeare/tiny_shakespeare_train.bin" };
	fs::path tiny_shakespeare_val{ "data/tinyshakespeare/tiny_shakespeare_val.bin" };
	fs::path train_tokens = exists(tiny_shakespeare_train) ? tiny_shakespeare_train : tiny_stories_train;
	fs::path val_tokens = exists(tiny_shakespeare_val) ? tiny_shakespeare_val : tiny_stories_val;
	fs::path checkpoint_path{ "data/gpt2_124M.bin" };
	fs::path tokenizer_path{ "data/gpt2_tokenizer.bin" };

	GPT2 gpt2{ checkpoint_path, B, T };
	DataLoader train_loader{ train_tokens, B, T, 0, 1, true };
	DataLoader val_loader{ val_tokens, B, T, 0, 1, false };

	INFO_PRINTLN("train dataset num_batches: {}", train_loader.num_tokens / (B * T));
	INFO_PRINTLN("val dataset num_batches: {}", val_loader.num_tokens / (B * T));

	Tokenizer tokenizer{ tokenizer_path };


	constexpr uint64_t rng_state = 1337;
	std::uniform_real_distribution<float> real_dist(0, 1);

	std::mt19937 shuffle_rng{ rng_state };
	Mati gen_tokens(B, T);

	INFO_PRINTLN("---------------{}---------------", fmt::styled("[Train...]", fmt::fg(fmt::color::green) | fmt::emphasis::bold));

	for (int step = 0; step <= iterations; ++step) {
		if (step % 10 == 0) {
			float val_loss = 0.0f;
			val_loader.Reset();
			for (int i = 0; i < val_num_batches; ++i) {
				val_loader.NextBatch();
				gpt2.Forward(val_loader.inputs, val_loader.targets);
				val_loss += gpt2.mean_loss;
			}
			val_loss /= val_num_batches;
			INFO_PRINTLN("val loss {}", val_loss);
		}
		if (step > 0 && step % 20 == 0) {
			gen_tokens.setConstant(tokenizer.eot_token);
			INFO_PRINTLN("Generating:\n---");
			for (int t = 1; t < genT; ++t) {
				gpt2.Forward(gen_tokens);
				float coin = real_dist(shuffle_rng);

				int next_token = SampleMult(gpt2.probs[0].row(t - 1), coin);
				gen_tokens(0, t) = next_token;
				if (tokenizer) {
					auto rep = tokenizer.decode(next_token);
					SafePrint(rep);
				} else {
					INFO_PRINT("{}", next_token);
				}
				std::fflush(stdout);
			}
			INFO_PRINTLN("");
		}
		auto start = std::chrono::steady_clock::now();
		train_loader.NextBatch();
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
		INFO_PRINTLN("step {}, mean loss:{}, duration:{}ms, forward duration:{}ms, backward duration:{}ms", step, gpt2.mean_loss, duration.count(), duration_f.count(), duration_b.count());
	}

	return 0;
}
