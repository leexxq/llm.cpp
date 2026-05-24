#include "dataloader.h"

#include "global.h"
#include "rand.h"

#include <fmt/ostream.h>

#include <Eigen/Core>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>

DataLoader::DataLoader(const fs::path &shards_dir, size_t B, size_t T,
		unsigned int process_rank,
		unsigned int num_processes,
		bool should_shuffle) : B_{ B },
							   T_{ T },
							   process_rank_{ process_rank },
							   num_processes_{ num_processes },
							   should_shuffle_{ should_shuffle },
							   total_batch_size_bytes_(num_processes * B * T * sizeof(uint16_t)),
							   local_batch_offset_bytes_(process_rank * B * T * sizeof(uint16_t)),
							   header_bytes_(kHeadSize * sizeof(int)) {
	if (fs::is_regular_file(shards_dir)) {
		shard_paths_.push_back(shards_dir);
	} else if (fs::is_directory(shards_dir)) {
		auto shards_iter = fs::directory_iterator{ shards_dir };
		for (const auto &entry : shards_iter) {
			if (!entry.is_regular_file()) {
				continue;
			}
			INFO_PRINTLN("find file : {}", fmt::streamed(entry.path()));
			shard_paths_.push_back(entry.path());
		}
	} else {
		INFO_PRINTLN("shard's directory is invalid: {}", fmt::streamed(shards_dir));
		exit(EXIT_FAILURE);
	}

	if (should_shuffle) {
		std::mt19937 shuffle_rng{ 42 + num_processes };
		this->shuffle_rng_ = shuffle_rng;
		this->shard_indices_.resize(shard_paths_.size());
		InitIdentityPermutation(shard_indices_);
		intra_shard_indices_.clear();
	}

	int64_t ntok_total = 0;
	for (int shard_index = 0; shard_index < shard_paths_.size(); shard_index++) {
		int64_t shard_ntok = LoadShard(shard_index);
		// we need at least one batch/shard, the way things are written right now.
		// can be relaxed a lot later.
		assert(shard_ntok >= (int64_t)(num_processes * B * T + 1));
		ntok_total += shard_ntok;
	}

	buffer_.resize(B * T + 1);
	inputs = Mati(B, T);
	targets = Mati(B, T);
	num_tokens = ntok_total;
	Reset();
}
int64_t DataLoader::LoadShard(int shard_index) {
	if (should_shuffle_) {
		shard_index = shard_indices_[shard_index];
	}

	fs::path filename = shard_paths_[shard_index];

	if (tokens_file_.is_open()) {
		tokens_file_.close();
	}

	tokens_file_ = std::ifstream{ filename, std::ios::binary };

	if (!tokens_file_.is_open()) {
		ERROR_PRINTLN("Error: cannot open file : {}", fmt::streamed(filename));
		exit(EXIT_FAILURE);
	}

	int header[kHeadSize];

	if (!tokens_file_.read(reinterpret_cast<char *>(header), kHeadSize * sizeof(int))) {
		ERROR_PRINTLN("Error: read tokens file: {} ", fmt::streamed(filename));
		exit(EXIT_FAILURE);
	}
	if (header[0] != 20240520) {
		ERROR_PRINTLN("Bad magic in the data file");
		INFO_PRINTLN("---> HINT: Are you passing in a correct file?");
		INFO_PRINTLN("---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.");
		exit(EXIT_FAILURE);
	}
	if (header[1] != 1) {
		ERROR_PRINTLN("Bad version in data file");
		exit(EXIT_FAILURE);
	}

	int64_t ntok = header[2]; // number of tokens in the file
	assert(ntok > 0); // we expect some tokens in the file. this should never trip, right?

	tokens_file_.seekg(0, std::ios::end);
	file_size_bytes_ = tokens_file_.tellg();
	// determine the file size and make sure it is consistent with the number of tokens
	tokens_file_.seekg(0, std::ios_base::beg); // seek back to the beginning
	// we expect ntok in the file to be consistent with filesize, assert that is the case
	int64_t expected_file_size = kHeadSize * sizeof(int) + ntok * sizeof(uint16_t);
	if (file_size_bytes_ != expected_file_size) {
		ERROR_PRINTLN("Error: file size is not as expected");
		exit(EXIT_FAILURE);
	}
	// -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
	shard_num_samples_ = (ntok * sizeof(uint16_t) - sizeof(uint16_t)) / total_batch_size_bytes_;

	INFO_PRINTLN("from file: {} , load tokens: {}", fmt::streamed(filename), ntok);
	return ntok;
}

void DataLoader::NextBatch() {
	if (current_sample_idx_ >= shard_num_samples_) {
		Advance();
	}

	LoadBatch();
	++current_sample_idx_;
}

void DataLoader::LoadBatch() {
	assert(!should_shuffle_ || (should_shuffle_ && !intra_shard_indices_.empty()));
	assert(current_sample_idx_ < shard_num_samples_);
	size_t idx = should_shuffle_ ? intra_shard_indices_[current_sample_idx_] : current_sample_idx_;
	size_t global_batch_offset_bytes = idx * total_batch_size_bytes_;
	int64_t current_offset = header_bytes_ + global_batch_offset_bytes + local_batch_offset_bytes_;

	size_t B = B_;
	size_t T = T_;
	// read B*T+1 uint16_t tokens from the file into buffer
	tokens_file_.seekg(current_offset, std::ios::beg);
	tokens_file_.read(reinterpret_cast<char *>(buffer_.data()), buffer_.size() * sizeof(uint16_t));
	// decode the buffer into inputs and targets (cast to int)
	for (int b = 0; b < B; ++b) {
		for (int t = 0; t < T; ++t) {
			int i = t + b * T;
			inputs(b, t) = static_cast<int>(buffer_[i]);
			targets(b, t) = static_cast<int>(buffer_[i + 1]);
		}
	}
	if (B > 1 && T > 1) {
		DEBUG_PRINTLN("inputs: \n{}...", fmt::streamed(inputs.block<2, 2>(0, 0)));
		DEBUG_PRINTLN("targets: \n{}...", fmt::streamed(targets.block<2, 2>(0, 0)));
	}
}

void DataLoader::Advance() {
	if (current_shard_idx_ == shard_paths_.size() - 1) {
		Reset();
		return;
	}

	current_shard_idx_ = (current_shard_idx_ + 1) % shard_paths_.size();
	current_sample_idx_ = 0;
	LoadShard(current_shard_idx_);

	if (should_shuffle_) {
		PrepareIntraShardIndices();
	}
}

void DataLoader::Resume(size_t shard_idx, size_t sample_idx) {
	current_sample_idx_ = sample_idx;
	current_shard_idx_ = shard_idx;
	LoadShard(current_shard_idx_);
}

void DataLoader::Reset() {
	current_shard_idx_ = 0;
	current_sample_idx_ = 0;

	if (should_shuffle_) { // shuffle the shards
		RandomPermutation(shard_indices_, shuffle_rng_);
	}

	LoadShard(current_shard_idx_);

	if (should_shuffle_) {
		PrepareIntraShardIndices();
	}
}

void DataLoader::PrepareIntraShardIndices() {
	// shuffle the examples inside the shards
	if (!intra_shard_indices_.empty()) {
		// in case shards have different number of samples / sizes
		intra_shard_indices_.clear();
	}
	intra_shard_indices_.resize(shard_num_samples_);
	InitIdentityPermutation(intra_shard_indices_);
	RandomPermutation(intra_shard_indices_, shuffle_rng_);
}
