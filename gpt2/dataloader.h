#pragma once
#include "global.h"


#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <vector>
namespace fs = std::filesystem;
using u16ptr = std::unique_ptr<uint16_t[]>;
using iptr = std::unique_ptr<int[]>;

class DataLoader {
private:
	constexpr static int kHeadSize = 256;
	// variables related to distributed training
	// each process/worker has to access different parts of the data
	unsigned int process_rank_;
	unsigned int num_processes_;
	// batch and token information
	size_t B_;
	size_t T_;
	size_t shard_num_samples_; // total number of samples in the current shard per process
	// shards and current position
	// glob_t glob_result; // stores the result of glob, for all shards we want to iterate
	size_t current_shard_idx_; // the current shard we are reading from
	size_t current_sample_idx_; // the current sample we are reading from
	// file handle
	std::ifstream tokens_file_;
	// data buffers
	std::vector<uint16_t> buffer_; // we fread data from file into this buffer
	// random shuffle related variables
	std::mt19937 shuffle_rng_;
	bool should_shuffle_;
	std::vector<int> shard_indices_;
	std::vector<int> intra_shard_indices_;

	std::vector<fs::path> shard_paths_;
	// sizes in bytes
	size_t total_batch_size_bytes_; // total across all processes
	size_t local_batch_offset_bytes_; // inner-sample offset for this process
	size_t header_bytes_; // header size in bytes
	int64_t file_size_bytes_;

public:
	size_t num_tokens; // total number of tokens
	Matf inputs; // input tokens into transformer
	Matf targets; // target tokens for the transformer

private:
	int64_t LoadShard(int shard_index);
	void PrepareIntraShardIndices();
	void Advance();

public:
	DataLoader() = delete;
	DataLoader(const DataLoader &) = delete;
	DataLoader(const DataLoader &&) = delete;
	explicit DataLoader(const fs::path &shard_dir, size_t B, size_t T,
			unsigned int process_rank,
			unsigned int num_processes,
			bool should_shuffle);
	~DataLoader() {}

	void LoadBatch();
	void NextBatch();
	void Resume(size_t shard_idx, size_t sample_idx);
	void Reset();
};
