#include "gpt2.h"
#include "log.h"

#include "adamw.h"
#include "encoder.h"
#include "fmt/ostream.h"
#include "global.h"
#include "layernorm.h"
#include "matmul.h"
#include "softmax.h"

#include <Eigen/Core>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <tuple>
void GPT2Config::Print() const {
	INFO_PRINTLN("---------------------------------");
	INFO_PRINTLN("{}: {}", fmt::styled("max_seq_len", fmt::fg(fmt::color::blue)), max_seq_len);
	INFO_PRINTLN("{}: {}", fmt::styled("vocab_size", fmt::fg(fmt::color::blue)), vocab_size);
	INFO_PRINTLN("{}: {}", fmt::styled("padded_vocab_size", fmt::fg(fmt::color::blue)), padded_vocab_size);
	INFO_PRINTLN("{}: {}", fmt::styled("num_layers", fmt::fg(fmt::color::blue)), num_layers);
	INFO_PRINTLN("{}: {}", fmt::styled("num_heads", fmt::fg(fmt::color::blue)), num_heads);
	INFO_PRINTLN("{}: {}", fmt::styled("channels", fmt::fg(fmt::color::blue)), channels);
	INFO_PRINTLN("---------------------------------");
}

void GPT2::Init(size_t B, size_t T) {
	INFO_PRINTLN("---------------{}---------------", fmt::styled("[GPT-2]", fmt::fg(fmt::color::green) | fmt::emphasis::bold));
	config_.Print();

	// Init GPT-2 Arch
	{
		encoder_ = Encoder{ config_.padded_vocab_size, config_.max_seq_len, config_.channels };

		//layers init
		for (int l = 0; l < config_.num_layers; ++l) {
			layers_.emplace_back(B, T, config_.channels, config_.vocab_size, config_.num_heads);
		}
		layernormf_ = LayerNorm{ B, T, config_.channels };
		loss_ = CrossEntropy{ B, T };
		// residual_ = VecLBTC(config_.num_layers);

		residual_ = makeZero(config_.num_layers, B, T, config_.channels);
		d_logits_ = makeZero(B, T, config_.padded_vocab_size);
		d_lnf_ = makeZero(B, T, config_.channels);
		d_residual3_ = makeZero(config_.num_layers, B, T, config_.channels);
		d_encoded_ = makeZero(B, T, config_.channels);
	}

	//params and grads data
	//will be used for updates
	{
		params_memory_.emplace_back(encoder_.wte.data(), encoder_.wte.size());
		grads_memory_.emplace_back(encoder_.d_wte.data(), encoder_.d_wte.size());

		params_memory_.emplace_back(encoder_.wpe.data(), encoder_.wpe.size());
		grads_memory_.emplace_back(encoder_.d_wpe.data(), encoder_.d_wpe.size());
		for (auto &layer : layers_) {
			params_memory_.emplace_back(layer.layernorm1.gamma.data(), layer.layernorm1.gamma.size());
			grads_memory_.emplace_back(layer.layernorm1.d_gamma.data(), layer.layernorm1.d_gamma.size());
			params_memory_.emplace_back(layer.layernorm1.beta.data(), layer.layernorm1.beta.size());
			grads_memory_.emplace_back(layer.layernorm1.d_beta.data(), layer.layernorm1.d_beta.size());

			params_memory_.emplace_back(layer.qkv.weight.data(), layer.qkv.weight.size());
			grads_memory_.emplace_back(layer.qkv.d_weight.data(), layer.qkv.d_weight.size());
			params_memory_.emplace_back(layer.qkv.bias.data(), layer.qkv.bias.size());
			grads_memory_.emplace_back(layer.qkv.d_bias.data(), layer.qkv.d_bias.size());

			params_memory_.emplace_back(layer.att_proj.weight.data(), layer.att_proj.weight.size());
			grads_memory_.emplace_back(layer.att_proj.d_weight.data(), layer.att_proj.d_weight.size());
			params_memory_.emplace_back(layer.att_proj.bias.data(), layer.att_proj.bias.size());
			grads_memory_.emplace_back(layer.att_proj.d_bias.data(), layer.att_proj.d_bias.size());

			params_memory_.emplace_back(layer.layernorm2.gamma.data(), layer.layernorm2.gamma.size());
			grads_memory_.emplace_back(layer.layernorm2.d_gamma.data(), layer.layernorm2.d_gamma.size());
			params_memory_.emplace_back(layer.layernorm2.beta.data(), layer.layernorm2.beta.size());
			grads_memory_.emplace_back(layer.layernorm2.d_beta.data(), layer.layernorm2.d_beta.size());

			params_memory_.emplace_back(layer.fch.weight.data(), layer.fch.weight.size());
			grads_memory_.emplace_back(layer.fch.d_weight.data(), layer.fch.d_weight.size());
			params_memory_.emplace_back(layer.fch.bias.data(), layer.fch.bias.size());
			grads_memory_.emplace_back(layer.fch.d_bias.data(), layer.fch.d_bias.size());

			params_memory_.emplace_back(layer.fcproj.weight.data(), layer.fcproj.weight.size());
			grads_memory_.emplace_back(layer.fcproj.d_weight.data(), layer.fcproj.d_weight.size());
			params_memory_.emplace_back(layer.fcproj.bias.data(), layer.fcproj.bias.size());
			grads_memory_.emplace_back(layer.fcproj.d_bias.data(), layer.fcproj.d_bias.size());
		}

		params_memory_.emplace_back(layernormf_.gamma.data(), layernormf_.gamma.size());
		grads_memory_.emplace_back(layernormf_.d_gamma.data(), layernormf_.d_gamma.size());
		params_memory_.emplace_back(layernormf_.beta.data(), layernormf_.beta.size());
		grads_memory_.emplace_back(layernormf_.d_beta.data(), layernormf_.d_beta.size());
	}

	//AdamW's m and v
	{
		size_t params_size = params_memory_.size();
		m_ = StdVec<Vecf>(params_size);
		v_ = StdVec<Vecf>(params_size);
		for (int i = 0; i < params_size; ++i) {
			m_[i] = Vecf::Zero(params_memory_[i].second);
			v_[i] = Vecf::Zero(params_memory_[i].second);
		}
	}
}

GPT2::GPT2(const std::filesystem::path &path, size_t B, size_t T) : checkpoint_path_(path), B_(B), T_(T) {
	std::ifstream f{ path, std::ios::binary };
	if (!f.is_open()) {
		ERROR_PRINTLN("Error: connot read file: {}", fmt::streamed(path));
		exit(EXIT_FAILURE);
	}

	int model_header[256];

	f.read(reinterpret_cast<char *>(model_header), sizeof(model_header));

	if (model_header[0] != 20240326) {
		ERROR_PRINTLN("Bad magic model file: {}", fmt::streamed(path));
		exit(EXIT_FAILURE);
	}
	if (model_header[1] != 3) {
		ERROR_PRINTLN("Bad version in model file: {}", fmt::streamed(path));
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

	Init(B_, T_);

	//Eigen default memory seq is column-major
	//but the data to read is row-major
	{
		Vecf tmp_row(config_.channels);
		for (int i = 0; i < config_.padded_vocab_size; ++i) {
			f.read(reinterpret_cast<char *>(tmp_row.data()), sizeof(float) * config_.channels);
			encoder_.wte.row(i) = tmp_row;
		}
		DEBUG_PRINTLN("load success wte : \n{}", fmt::streamed(encoder_.wte.block<3, 3>(0, 0)));

		for (int i = 0; i < config_.max_seq_len; ++i) {
			f.read(reinterpret_cast<char *>(tmp_row.data()), sizeof(float) * config_.channels);
			encoder_.wpe.row(i) = tmp_row;
		}
		DEBUG_PRINTLN("load success wpe : \n{}", fmt::streamed(encoder_.wpe.block<3, 3>(0, 0)));
	}

	for (int i = 0; i < config_.num_layers; ++i) {
		auto ln1w = layers_[i].layernorm1.gamma.data();
		f.read(reinterpret_cast<char *>(ln1w), sizeof(float) * config_.channels);
		// DEBUG_PRINTLN("load success layer {} ln1w : \n{}", i + 1, fmt::streamed(layers_[i].layernorm1.gamma.segment(0, 9)));
	}
	//
	for (int i = 0; i < config_.num_layers; ++i) {
		auto ln1b = layers_[i].layernorm1.beta.data();
		f.read(reinterpret_cast<char *>(ln1b), sizeof(float) * config_.channels);
		// DEBUG_PRINTLN("load success layer {} ln1b : \n{}", i + 1, fmt::streamed(layers_[i].layernorm1.beta.segment(0, 9)));
	}

	// due to eigen matrix default column-major 
	// and our matrix's shape is (C,OC) ,but the file to read is row-major and shape is (OC,C)
	// so does'nt necessary to convert ;
	for (int i = 0; i < config_.num_layers; ++i) {
		auto qkvw = layers_[i].qkv.weight.data();
		f.read(reinterpret_cast<char *>(qkvw), sizeof(float) * config_.channels * 3 * config_.channels);
	}

	for (int i = 0; i < config_.num_layers; ++i) {
		auto qkvb = layers_[i].qkv.bias.data();
		f.read(reinterpret_cast<char *>(qkvb), sizeof(float) * 3 * config_.channels);
	}

	for (int i = 0; i < config_.num_layers; ++i) {
		auto att_projw = layers_[i].att_proj.weight.data();
		f.read(reinterpret_cast<char *>(att_projw), sizeof(float) * config_.channels * config_.channels);
	}

	for (int i = 0; i < config_.num_layers; ++i) {
		auto att_projb = layers_[i].att_proj.bias.data();
		f.read(reinterpret_cast<char *>(att_projb), sizeof(float) * config_.channels);
	}

	for (int i = 0; i < config_.num_layers; ++i) {
		auto ln2w = layers_[i].layernorm2.gamma.data();
		f.read(reinterpret_cast<char *>(ln2w), sizeof(float) * config_.channels);
	}

	for (int i = 0; i < config_.num_layers; ++i) {
		auto ln2b = layers_[i].layernorm2.beta.data();
		f.read(reinterpret_cast<char *>(ln2b), sizeof(float) * config_.channels);
	}

	for (int i = 0; i < config_.num_layers; ++i) {
		auto fcw = layers_[i].fch.weight.data();
		f.read(reinterpret_cast<char *>(fcw), sizeof(float) * 4 * config_.channels * config_.channels);
	}

	for (int i = 0; i < config_.num_layers; ++i) {
		auto fcb = layers_[i].fch.bias.data();
		f.read(reinterpret_cast<char *>(fcb), sizeof(float) * 4 * config_.channels);
	}

	for (int i = 0; i < config_.num_layers; ++i) {
		auto fcprojw = layers_[i].fcproj.weight.data();
		f.read(reinterpret_cast<char *>(fcprojw), sizeof(float) * 4 * config_.channels * config_.channels);
	}
	DEBUG_PRINTLN("load success end layers fcprojw: \n{}", fmt::streamed(layers_.back().fcproj.weight.block<3, 3>(0, 0)));

	for (int i = 0; i < config_.num_layers; ++i) {
		auto fcprojb = layers_[i].fcproj.bias.data();
		f.read(reinterpret_cast<char *>(fcprojb), sizeof(float) * config_.channels);
	}

	f.read(reinterpret_cast<char *>(layernormf_.gamma.data()), sizeof(float) * config_.channels);
	f.read(reinterpret_cast<char *>(layernormf_.beta.data()), sizeof(float) * config_.channels);

	DEBUG_PRINTLN("load success lnfb: \n{}", fmt::streamed(layernormf_.beta.tail(3)));
}

void GPT2::Forward(StdVec<int> &inputs, StdVec<int> &targets) {
	using MapRow = Eigen::Map<Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>;
	assert(B_ * T_ == inputs.size());
	this->inputs_ = MapRow(inputs.data(),B_,T_);
	if(!targets.empty()){
		this->targets_ = MapRow(targets.data(),B_,T_); 
	}

	DEBUG_PRINTLN("---------------{}---------------", fmt::styled("[Forward]", fmt::fg(fmt::color::green) | fmt::emphasis::bold));
	encoded_ = encoder_.Forward(inputs_);

	DEBUG_PRINTLN("encoder forward sharp: ({} ,{} , {})", encoded_.size(), encoded_[1].rows(), encoded_[0].cols());
	DEBUG_PRINTLN("enc[0] : \n{}", fmt::streamed(encoded_[0].block<2, 2>(0, 0)));

	DEBUG_PRINTLN("--------------{}:{}--------------------", fmt::styled("Layer", fmt::fg(fmt::color::green) | fmt::emphasis::bold), 1);
	residual_.front() = layers_.front().Forward(encoded_);
	DEBUG_PRINTLN("residual_ first : \n{}", fmt::streamed(residual_.front()[0].block<3, 3>(0, 0)));

	for (int l = 1; l < config_.num_layers; ++l) {
		DEBUG_PRINTLN("--------------{}:{}--------------------", fmt::styled("Layer", fmt::fg(fmt::color::green) | fmt::emphasis::bold), l + 1);
		residual_[l] = layers_[l].Forward(residual_[l - 1]);
	}

	lnf_ = layernormf_.Forward(residual_.back());
	DEBUG_PRINTLN("lnf forward sharp: ({} , {} , {})", lnf_.size(), lnf_[0].rows(), lnf_[0].cols());
	DEBUG_PRINTLN("lnf[0]: \n{}", fmt::streamed(lnf_[0].block<2, 2>(0, 0)));

	logits_ = MatMulForward(lnf_, encoder_.wte.transpose());
	DEBUG_PRINTLN("logits forward sharp: ({} , {} , {})", logits_.size(), logits_[0].rows(), logits_[0].cols());
	DEBUG_PRINTLN("logits[0]: \n{}", fmt::streamed(logits_[0].block<2, 2>(0, 0)));

	probs = makeVecBTV(B_,T_,config_.vocab_size);
	assert(logits_.front().cols() >= config_.vocab_size);
	for (int b = 0; b < B_; ++b) {
		probs[b] = softmax(logits_[b].block(0, 0, T_, config_.vocab_size));
	}
	DEBUG_PRINTLN("probs[0] row 0 sum = {}", probs[0].row(0).sum());
	DEBUG_PRINTLN("probs forward sharp: ({} , {} , {})", probs.size(), probs[0].rows(), probs[0].cols());
	DEBUG_PRINTLN("probs[0]: \n{}", fmt::streamed(probs[0].block<9, 9>(0, 0)));

	this->mean_loss = targets_.size() > 0 ? loss_.Forward(probs, targets_) : -1.0f;

	DEBUG_PRINTLN("losses 3 first:\n {}", fmt::streamed(loss_.losses[0].head(3)));

	DEBUG_PRINTLN("{} : {}", fmt::styled("mean loss", fmt::fg(fmt::color::dark_red)), mean_loss);

	DEBUG_PRINTLN("{}", fmt::styled("Forward finished!", fmt::fg(fmt::color::blue)));
}

void GPT2::Backward() {
	DEBUG_PRINTLN("---------------{}---------------", fmt::styled("[Backward]", fmt::fg(fmt::color::green) | fmt::emphasis::bold));
	CrossEntropySoftmaxBackward(d_logits_, probs, targets_);
	DEBUG_PRINTLN("d_logits_ cross entropy softmax backward sharp: ({} , {} , {})", d_logits_.size(), d_logits_[0].rows(), d_logits_[0].cols());
	DEBUG_PRINTLN("d_logits_[0]: \n{}", fmt::streamed(d_logits_[0].block<3, 3>(0, 0)));
	size_t s_b, s_t, s_c;

	MatMulBackward(d_logits_, lnf_, encoder_.wte, d_lnf_, encoder_.d_wte, true, B_, T_, config_.channels, config_.vocab_size);

	DEBUG_PRINTLN("d_lnf_ matmul backward sharp: ({} , {} , {})", d_lnf_.size(), d_lnf_[0].rows(), d_lnf_[0].cols());
	DEBUG_PRINTLN("d_lnf_[0]: \n{}", fmt::streamed(d_lnf_[0].block<2, 2>(0, 0)));
	DEBUG_PRINTLN("d_wte: \n{}", fmt::streamed(encoder_.d_wte.block<2, 2>(0, 0)));
	DEBUG_PRINTLN("d_wpe: \n{}", fmt::streamed(encoder_.d_wpe.block<2, 2>(0, 0)));

	d_residual3_.back() += layernormf_.Backward(d_lnf_, residual_.back());

	std::tie(s_b, s_t, s_c) = GetShape(d_residual3_.back());
	DEBUG_PRINTLN("d_residual3_ last shape ({},{},{})", s_b, s_t, s_c);
	DEBUG_PRINTLN("d_residual3_ last[0]: \n{}", fmt::streamed(d_residual3_.back()[0].block<2, 2>(0, 0)));

	for (int l = config_.num_layers - 1; l > 0; --l) {
		DEBUG_PRINTLN("--------------{}:{}--------------------", fmt::styled("Layer", fmt::fg(fmt::color::green) | fmt::emphasis::bold), l + 1);
		d_residual3_[l - 1] += layers_[l].Backward(d_residual3_[l], residual_[l - 1]);
	}

	DEBUG_PRINTLN("--------------{}:{}--------------------", fmt::styled("Layer", fmt::fg(fmt::color::green) | fmt::emphasis::bold), 1);
	d_encoded_ += layers_.front().Backward(d_residual3_[0], encoded_);
	std::tie(s_b, s_t, s_c) = GetShape(d_encoded_);
	DEBUG_PRINTLN("d_encoded shape ({},{},{})", s_b, s_t, s_c);
	DEBUG_PRINTLN("d_encoded last[0]: \n{}", fmt::streamed(d_encoded_[0].block<2, 2>(0, 0)));

	encoder_.Backward(d_encoded_, this->inputs_);
	DEBUG_PRINTLN("d_wte: \n{}", fmt::streamed(encoder_.d_wte.block<2, 2>(0, 0)));
	DEBUG_PRINTLN("d_wpe: \n{}", fmt::streamed(encoder_.d_wpe.block<2, 2>(0, 0)));

	DEBUG_PRINTLN("{}", fmt::styled("Backward finished!", fmt::fg(fmt::color::blue)));
}

void SetZero(VecBTC &btc) {
	for (auto &b : btc) {
		b.setZero();
	}
}

void SetZero(VecLBTC &lbtc) {
	for (auto &l : lbtc) {
		SetZero(l);
	}
}

void SetZero(Matf &mat) {
	mat.setZero();
}

void SetZero(Vecf &vec) {
	vec.setZero();
}

void ZeroGrad(Encoder &encoder) {
	SetZero(encoder.d_wte);
	SetZero(encoder.d_wpe);
}

void ZeroGrad(LayerNorm &layernorm) {
	SetZero(layernorm.d_gamma);
	SetZero(layernorm.d_beta);
}

void ZeroGrad(MatMul &matmul) {
	SetZero(matmul.d_bias);
	SetZero(matmul.d_weight);
}

void ZeroGrad(Layer &l) {
	ZeroGrad(l.layernorm1);
	ZeroGrad(l.layernorm2);
	ZeroGrad(l.fch);
	ZeroGrad(l.fcproj);
	ZeroGrad(l.qkv);
	ZeroGrad(l.att_proj);

	SetZero(l.dl_residual2);
	SetZero(l.dl_fcproj);
	SetZero(l.dl_fch_gelu);
	SetZero(l.dl_fch);
	SetZero(l.dl_ln2);
	SetZero(l.dresidual);
	SetZero(l.dl_attproj);
	SetZero(l.dl_atty);
	SetZero(l.dl_qkv);
	SetZero(l.dl_ln1);
}

void ZeroGrad(StdVec<Layer> &layers) {
	for (auto &layer : layers) {
		ZeroGrad(layer);
	}
}

void GPT2::ZeroGrad() {
	::SetZero(d_logits_);
	::SetZero(d_lnf_);
	::SetZero(d_residual3_);
	::SetZero(d_encoded_);
	::ZeroGrad(encoder_);
	::ZeroGrad(layers_);
	::ZeroGrad(layernormf_);
}

void GPT2::Update(float lr, float beta1, float beta2, float eps, float weight, int t) {
	AdamWParams adamw_params{
		lr,
		beta1,
		beta2,
		eps,
		weight,
		t
	};

	size_t params_size = params_memory_.size();
	for (int i = 0; i < params_size; ++i) {
		auto [data, size] = params_memory_[i];
		auto [grads_data, grads_size] = grads_memory_[i];
		assert(size == grads_size && size == m_[i].size());
		AdamW(data, grads_data, m_[i].data(), v_[i].data(), m_[i].size(), adamw_params);
	}

	DEBUG_PRINTLN("{}", fmt::styled("Update Successfully!", fmt::fg(fmt::color::blue)));
}
