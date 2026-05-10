#include "dataloader.h"

#include "rand.h"

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
			std::cout << "find file:" << entry.path() << std::endl;
			shard_paths_.push_back(entry.path());
		}
	} else {
		std::cerr << "shards directory is invalid: " << shards_dir << std::endl;
		exit(EXIT_FAILURE);
	}

	if (should_shuffle) {
		std::mt19937 shuffle_rng{ 42 + num_processes };
		std::uniform_int_distribution<int> dist(1, 2);
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
	inputs = Mat(B, T);
	targets = Mat(B, T);
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
		std::cerr << "Error: cannot open file : " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	int header[kHeadSize];

	if (!tokens_file_.read(reinterpret_cast<char *>(header), kHeadSize * sizeof(int))) {
		std::cerr << "Error: read tokens file: " << filename << std::endl;
		exit(EXIT_FAILURE);
	}
	if (header[0] != 20240520) {
		std::cerr << ("Bad magic in the data file\n");
		std::cout << ("---> HINT: Are you passing in a correct file?\n");
		std::cout << ("---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.\n");
		exit(EXIT_FAILURE);
	}
	if (header[1] != 1) {
		std::cerr << ("Bad version in data file\n");
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
		std::cerr << "Error: file size is not as expected" << std::endl;
		exit(EXIT_FAILURE);
	}
	// -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
	shard_num_samples_ = (ntok * sizeof(uint16_t) - sizeof(uint16_t)) / total_batch_size_bytes_;

	std::cout << "from file: " << filename << " load tokens: " << ntok << std::endl;
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
	std::cout << "inputs : " << std::endl;
	std::cout << inputs.block<2, 2>(0, 0) << "\n"
			  << ".\n"
			  << ".\n"
			  << "." << std::endl;
	std::cout << "targets: " << std::endl;
	std::cout << targets.block<2, 2>(0, 0) << "\n"
			  << ".\n"
			  << ".\n"
			  << "." << std::endl;
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
