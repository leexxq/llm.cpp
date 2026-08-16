#include "CLI/CLI.hpp"
#include "cuda/error.cuh"
#include "gpt2/cuda/gpt2cuda.h"
#include "utils/dataloader.h"
#include "utils/tokenizer.h"
#include "gpt2/log.h"
#include "cutlass/util/GPU_Clock.hpp"

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <random>
#include <string>
using namespace gpt2cuda;


void SafePrint(const std::string &rep) {
	if (rep.empty()) {
		return;
	}
	if (!(std::isprint(rep[0]) || std::isspace(rep[0]))) {
		return;
	}
	INFO_PRINT("{}", rep);
}

struct GenContext{
	public:
		float* coins;
		int* gen_context;//(B,T)
		Tokenizer* tokenizer;
		std::size_t vocab_size;
		std::size_t next_token_idx;
		std::size_t batch_size;
		std::size_t seq_len;
		float* probs;//(B,T,Vp)
	public:
		GenContext(){};
		GenContext(int * gen_context,size_t start_idx,float * probs,float* coins,Tokenizer* tokenizer,size_t B,size_t T,size_t probs_size){
			this->gen_context = gen_context;
			this->coins = coins;
			this->next_token_idx = start_idx;
			this->probs = probs;
			this->batch_size = B;
			this->seq_len = T;
			this->vocab_size = probs_size / (B * T);
			this->tokenizer = tokenizer;
		};
};

void CUDART_CB on_probs_ready(void *userData) {
    GenContext* ctx = reinterpret_cast<GenContext*>(userData);
    float* probs = ctx->probs;
    int* gen_context = ctx->gen_context;
    std::size_t vocab_size = ctx->vocab_size;
    std::size_t batch_size = ctx->batch_size;
    std::size_t next_token_idx = ctx->next_token_idx;
    std::size_t seq_len = ctx->seq_len;

    for(int b= 0 ; b < batch_size; ++b){
        float cdf = 0.f;
		gen_context[b*seq_len + next_token_idx] = vocab_size - 1; 
        for (int i = 0; i < vocab_size; ++i) {
            cdf += probs[b *seq_len * vocab_size + (next_token_idx -1 )* vocab_size + i];
            if (ctx->coins[b] < cdf) {
                gen_context[b*seq_len + next_token_idx] = i;
				break;
            }
        }
    }
}

void CUDART_CB on_token_ready(void * userData){
    GenContext* ctx = reinterpret_cast<GenContext*>(userData);
	int next_token = ctx->gen_context[ctx->next_token_idx];
	Tokenizer& tokenizer = *(ctx->tokenizer);
	if (tokenizer) {
		auto rep = tokenizer.decode(next_token);
		SafePrint(rep);
	} else {
		INFO_PRINT("{}", next_token);
	}
	std::fflush(stdout);
}

template<class RandomEngine>
void GenerateText(StdVeci& gen_tokens,Tokenizer& tokenizer,gpt2cuda::GPT2& gpt2,size_t genT,size_t B,size_t T,RandomEngine re,cudaStream_t stream){
	INFO_PRINTLN("Generating:\n---");
	PinVecf probs(gpt2.GetProbsSize());
	StdVec<GenContext> contexts;
	contexts.reserve(genT - 1);
	StdVec<StdVecf> coins(genT-1);
	cudaEvent_t gen_finish_event;
	CUDA_CHECK(cudaEventCreate(&gen_finish_event));
	std::uniform_real_distribution<float> real_dist(0, 1);
	for (size_t t = 1; t < genT; ++t) {
		for(int i =0 ; i < B ; ++i){
			coins[t-1].emplace_back(real_dist(re));
		}
		contexts.emplace_back(gen_tokens.data(),t,probs.data(),coins[t-1].data(),&tokenizer,B,T,gpt2.GetProbsSize());
		gpt2.Forward(gen_tokens,stream);
		gpt2.GetProbs(probs, stream);
		CUDA_CHECK(cudaLaunchHostFunc(stream, on_probs_ready, &contexts.back()));
		CUDA_CHECK(cudaLaunchHostFunc(stream, on_token_ready, &contexts.back()));
	}
	CUDA_CHECK(cudaEventRecord(gen_finish_event));
	CUDA_CHECK(cudaEventSynchronize(gen_finish_event));
	INFO_PRINTLN("");

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
	bool one_batch = false;
	app.add_option("-b,--batch",B,"training batchs");
	app.add_option("-t,--seq_length",T,"train tokens length");
	app.add_option("--gen_token",genT,"generative text's tokens length");
	app.add_option("--val_sets_batch",val_num_batches,"validate sets's batch for test");
	app.add_option("-i,--iteration",iterations,"train iterations number");

	app.add_flag("--one_batch",one_batch,"repeat train one batch");

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

	gpt2cuda::GPT2 gpt2{ checkpoint_path, B, T };
	DataLoader train_loader{ train_tokens, B, T, 0, 1, true };
	DataLoader val_loader{ val_tokens, B, T, 0, 1, false };

	INFO_PRINTLN("train dataset num_batches: {}", train_loader.num_tokens / (B * T));
	INFO_PRINTLN("val dataset num_batches: {}", val_loader.num_tokens / (B * T));

	Tokenizer tokenizer{ tokenizer_path };


	constexpr uint64_t rng_state = 1337;

	std::mt19937 shuffle_rng{ rng_state };
	gpt2cuda::StdVec<int> gen_tokens(B*T);

	INFO_PRINTLN("---------------{}---------------", fmt::styled("[Train...]", fmt::fg(fmt::color::green) | fmt::emphasis::bold));

	StdVec<int> inputs(B*T);
	StdVec<int> targets(B*T);
	bool next_batch = true;
	int log_iterations = 10;
	
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	float duration = 0;
	float duration_f = 0;
	float duration_b = 0;
	float duration_u = 0;

	for (int step = 0; step <= iterations; ++step) {
		if (step % 20 == 0) {
			gpt2.ZeroLoss(stream);
			float val_loss = 0.0f;
			val_loader.Reset();
			for (int i = 0; i < val_num_batches; ++i) {
				val_loader.NextBatch(inputs,targets);
				gpt2.Forward(inputs,targets,stream);
			}
			
			val_loss = gpt2.GetLoss(stream) / val_num_batches;
			INFO_PRINTLN("val loss {}", val_loss);
			gpt2.ZeroLoss(stream);
		}
		if (step > 0 && step % 20 == 0) {
			gen_tokens.assign(B*T,tokenizer.eot_token);
			gpt2.ZeroLoss(stream);
			GenerateText(gen_tokens, tokenizer, gpt2, genT, B, T, shuffle_rng, stream);
			gpt2.ZeroLoss(stream);
		}

		gpt2.ZeroLoss(stream);
		
		GPU_Clock timer;
		GPU_Clock timer_forward;
		GPU_Clock timer_backward;
		GPU_Clock timer_update;
		timer.start();
		
		if(next_batch){
			train_loader.NextBatch(inputs,targets);
		}
		if(one_batch){
			next_batch = false;
		}

		timer_forward.start();
		gpt2.Forward(inputs,targets,stream);
		duration_f += timer_forward.milliseconds();

		gpt2.ZeroGrad(stream);

		timer_backward.start();
		gpt2.Backward(stream);
		duration_b += timer_backward.milliseconds();

		timer_update.start();
		gpt2.Update(1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step + 1,stream);
		duration_u += timer_update.milliseconds();

		duration += timer.milliseconds();
		
		
		if(step > 0 && step % log_iterations == 0){
			duration/= log_iterations;
			duration_f/= log_iterations;
			duration_b/= log_iterations;
			duration_u/= log_iterations;
			float mean_loss = gpt2.GetLoss(stream) ;

			INFO_PRINTLN("step {}, mean loss:{}, duration:{}ms, forward duration:{}ms, backward duration:{}ms , update duration:{}ms", step, mean_loss, duration, duration_f, duration_b,duration_u);

			duration = 0;
			duration_f = 0;
			duration_b = 0;
			duration_u = 0;
		}
	}
	CUDA_CHECK(cudaStreamDestroy(stream));

	return 0;
}