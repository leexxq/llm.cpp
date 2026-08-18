#include "CLI/CLI.hpp"
#include "cuda/error.cuh"
#include "cuda/pinvector.cuh"
#include "gpt2/cuda/gpt2cuda.h"
#include "utils/dataloader.h"
#include "utils/tokenizer.h"
#include "gpt2/log.h"
#include "cuda/gpu_clocker.cuh"

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <ratio>
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
	private:
		GenContext(){};
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
void GenerateText(PinVeci& gen_tokens,Tokenizer& tokenizer,gpt2cuda::GPT2& gpt2,size_t genT,size_t B,size_t T,RandomEngine re,GPT2::Stream& stream){
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
		gpt2.SetTrainData(stream, gen_tokens);
		gpt2.Forward(stream);
		gpt2.GetProbs(probs, stream);
		CUDA_CHECK(cudaLaunchHostFunc(stream.GetStream(), on_probs_ready, &contexts.back()));
		CUDA_CHECK(cudaLaunchHostFunc(stream.GetStream(), on_token_ready, &contexts.back()));
	}
	CUDA_CHECK(cudaEventRecord(gen_finish_event));
	CUDA_CHECK(cudaEventSynchronize(gen_finish_event));
	INFO_PRINTLN("");

}


struct TrainArgs{
	size_t B = 4;
	size_t T = 64;
	size_t genT = 64;
	size_t val_num_batches = 5;
	size_t iterations = 40;
	size_t log_iterations = 10;
	size_t val_iterations = 20;
	size_t gen_iterations = 20;
	bool one_batch = false;
	bool enable_gen = false;
	bool enable_val = false;
	std::string datasets_dir;
};


int main(int argc, char **argv) {

	CLI::App app{"Train parameters"};
	TrainArgs args;
	app.add_option("-d,--datasets",args.datasets_dir,"datasets directory");

	app.add_option("-b,--batch",args.B,"training batchs");
	app.add_option("-t,--seq_length",args.T,"train tokens length");
	app.add_option("--gen_token",args.genT,"generative text's tokens length");
	app.add_option("--val_sets_batch",args.val_num_batches,"validate sets's batch for test");
	app.add_option("-i,--iteration",args.iterations,"train iterations number");
	app.add_flag("--one_batch",args.one_batch,"repeat train one batch");
	app.add_flag("--enable_gen",args.enable_gen,"start generate text after gen_iterations iterations");
	app.add_flag("--enable_val",args.enable_val,"start test valiadate sets on ");
	app.add_option("--gen_iterations",args.gen_iterations,"set generate iterations");
	app.add_option("--val_iterations",args.val_iterations,"set test iterations");

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

	gpt2cuda::GPT2 gpt2{ checkpoint_path, args.B, args.T };
	DataLoader train_loader{ train_tokens, args.B, args.T, 0, 1, true };
	DataLoader val_loader{ val_tokens, args.B, args.T, 0, 1, false };

	INFO_PRINTLN("train dataset num_batches: {}", train_loader.num_tokens / (args.B * args.T));
	INFO_PRINTLN("val dataset num_batches: {}", val_loader.num_tokens / (args.B * args.T));

	Tokenizer tokenizer{ tokenizer_path };


	constexpr uint64_t rng_state = 1337;

	std::mt19937 shuffle_rng{ rng_state };
	gpt2cuda::PinVeci gen_tokens(args.B*args.T);

	INFO_PRINTLN("---------------{}---------------", fmt::styled("[Train...]", fmt::fg(fmt::color::green) | fmt::emphasis::bold));

	PinVeci inputs(args.B*args.T);
	PinVeci targets(args.B*args.T);
	int log_iterations = 10;

	
	auto stream1 = gpt2.CreateStream();
	auto stream2 = gpt2.CreateStream();



	train_loader.NextBatch(inputs.begin(),targets.begin());
	gpt2.SetTrainData(stream1,inputs, targets);
	// GPUClock timer;
	
	
    auto start = std::chrono::high_resolution_clock::now();	
    auto end = std::chrono::high_resolution_clock::now();	
	for (int step = 0; step < args.iterations; ++step) {

		
		gpt2cuda::GPT2::Stream& cur_stream = step % 2==0 ? stream1 : stream2;
		gpt2cuda::GPT2::Stream& next_stream = step % 2== 0 ? stream2 : stream1;

		if (args.enable_val && step % args.val_iterations == 0) {
			gpt2.ZeroLoss(cur_stream);
			float val_loss = 0.0f;
			val_loader.Reset();
			for (int i = 0; i < args.val_num_batches; ++i) {
				val_loader.NextBatch(inputs.begin(),targets.begin());
				gpt2.SetTrainData(cur_stream, inputs,targets);
				gpt2.Forward(cur_stream);
			}
			val_loss = gpt2.GetLossSync(cur_stream) / args.val_num_batches;
			INFO_PRINTLN("val loss {}", val_loss);
		}
		

		gpt2.ZeroLoss(cur_stream);

		gpt2.Forward(cur_stream);

		gpt2.ZeroGrad(cur_stream);

		gpt2.Backward(cur_stream);

		gpt2.Update(1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step + 1,cur_stream);


		if(!args.one_batch){
			train_loader.NextBatch(inputs.begin(),targets.begin());
		}		

		gpt2.SetTrainData(next_stream,inputs, targets);
		
		if((step + 1) % args.log_iterations == 0 ){
			end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double,std::milli> elapsed = end -start;
			float duration = elapsed.count();
			float mean_loss = gpt2.GetLossSync(cur_stream) ;
			INFO_PRINTLN("step {}, mean loss:{}, duration:{}ms", step + 1, mean_loss, duration);
			start = std::chrono::high_resolution_clock::now();

		}
		
		if (args.enable_gen&&(step+1) % args.gen_iterations == 0) {
			gen_tokens.assign(args.B*args.T,tokenizer.eot_token);
			cur_stream.SetOnlyRefer(true);
			gpt2.SetTrainData(cur_stream,gen_tokens);
			GenerateText(gen_tokens, tokenizer, gpt2, args.genT, args.B, args.T, shuffle_rng, cur_stream);
			cur_stream.SetOnlyRefer(false);
		}
	}


	return 0;
}