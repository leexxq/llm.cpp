#include "cuda/softmax.cuh"
#include "layernorm.cuh"
#include "log.h"
#include "matmul.cuh"
#include "cross_entropy.cuh"
#include "encoder.cuh"
#include "gpt2cuda.h"
#include "adamw.h"
// #include "adamw.cuh"
#include <cassert>
#include <iostream>
#include <numeric>
#include <optional>
#include <fstream>
namespace gpt2cuda{

void GPT2Config::Print() const {
    std::cout << "max_seq_len:" << max_seq_len<<std::endl;
    std::cout << "vocab_size:" << vocab_size<<std::endl;
    std::cout << "padded_vocab_size:" <<padded_vocab_size<<std::endl;
    std::cout << "num_layers:" << num_layers<<std::endl;
    std::cout << "num_heads:" << num_heads<<std::endl;
    std::cout << "channels:" << channels<<std::endl;
}


GPT2::GPT2(const std::filesystem::path &path, size_t B, size_t T) : checkpoint_path_(path), B_(B), T_(T) {
	std::ifstream f{ path, std::ios::binary };
	if (!f.is_open()) {
        std::cerr << "Error: connot read file: " << path<<std::endl;
		exit(EXIT_FAILURE);
	}

	int model_header[256];

    using byte = char*;
	f.read(reinterpret_cast<byte>(model_header), sizeof(model_header));

	if (model_header[0] != 20240326) {
        std::cerr << "Bad magic model file: " << path<<std::endl;
		exit(EXIT_FAILURE);
	}
	if (model_header[1] != 3) {
        std::cerr << "Bad version in model file: " << path<<std::endl;
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


    Init(B, T);

    /**
    Unlike the CPU implementation, our default layout here is row-major, 
    and the shape of the `weight` parameter for matrix multiplication is (OC, C); 
    this means our matrix multiplication adheres to PyTorch's conventions.
    for details:https://docs.pytorch.org/docs/2.12/generated/torch.nn.Linear.html
    */


    try {
        //wte(Vp,C) row-major
        f.read(reinterpret_cast<byte>(wte_.data()),sizeof(float)*C * Vp);
        std::cout << "load success wte"  << std::endl;
        //wte(MaxT,C)
        f.read(reinterpret_cast<byte>(wpe_.data()),sizeof(float)*C * maxT);
        std::cout << "load success wpe"  << std::endl;

        for(int i =0 ; i < L; ++i){
            f.read(reinterpret_cast<byte>(layers_[i].l_ln1_gamma.data()),sizeof(float)*C );
        }
        for(int i =0 ; i < L; ++i){
            f.read(reinterpret_cast<byte>(layers_[i].l_ln1_beta.data()),sizeof(float)*C );
        }

        for(int i = 0 ; i < L ; ++i){
            f.read(reinterpret_cast<byte>(layers_[i].l_qkv_weight.data()),sizeof(float)*C* 3 *C );
        }
        for(int i = 0 ; i < L ; ++i){
            f.read(reinterpret_cast<byte>(layers_[i].l_qkv_bias.data()),sizeof(float) * 3*C );
        }
        for(int i = 0 ; i < L ; ++i){
            f.read(reinterpret_cast<byte>(layers_[i].l_attproj_weight.data()),sizeof(float) * C*C );
        }
        for(int i = 0 ; i < L ; ++i){
            f.read(reinterpret_cast<byte>(layers_[i].l_attproj_bias.data()),sizeof(float) * C );
        }
        for(int i = 0 ; i < L ; ++i){
            f.read(reinterpret_cast<byte>(layers_[i].l_ln2_gamma.data()),sizeof(float) * C );
        }
        for(int i = 0 ; i < L ; ++i){
            f.read(reinterpret_cast<byte>(layers_[i].l_ln2_beta.data()),sizeof(float) * C );
        }
        for(int i = 0 ; i < L ; ++i){
            f.read(reinterpret_cast<byte>(layers_[i].l_fch_weight.data()),sizeof(float) * C * 4*C );
        }
        for(int i = 0 ; i < L ; ++i){
            f.read(reinterpret_cast<byte>(layers_[i].l_fch_bias.data()),sizeof(float) *4*C );
        }
        for(int i = 0 ; i < L ; ++i){
            f.read(reinterpret_cast<byte>(layers_[i].l_fcproj_weight.data()),sizeof(float) *C*4*C );
        }
        for(int i = 0 ; i < L ; ++i){
            f.read(reinterpret_cast<byte>(layers_[i].l_fcproj_bias.data()),sizeof(float) *C);
        }

        f.read(reinterpret_cast<byte>(lnf_gamma_.data()),sizeof(float) *C);
        f.read(reinterpret_cast<byte>(lnf_beta_.data()),sizeof(float) *C);

        std::cout <<"load success lnf_beta"<<std::endl;
        std::cout << "load params MB: " << (params_bytes_ / (1024.0 * 1024.0)) << std::endl; 
        std::cout << "load optim params MB: " << (optimizer_bytes_ / (1024.0 * 1024.0)) << std::endl; 
    } catch (...) {
        f.close();
    }
}

void GPT2::Init(size_t B,size_t T){
    config_.Print();

    params_bytes_ = 0;
    inputs_ = makePinVec<int>({B,T});
    targets_ = makePinVec<int>({B,T});
    params_bytes_ += inputs_.size() * sizeof(int);
    params_bytes_ += targets_.size() * sizeof(int);

    {
        //encoder init

        encoded_ = makePinVecf({B,T,config_.channels});
        params_bytes_ += encoded_.size() * sizeof(float);

        //layers init
		for (int l = 0; l < config_.num_layers; ++l) {
			layers_.emplace_back(B, T, config_.channels, config_.vocab_size, config_.num_heads);
            params_bytes_ += layers_.back().GetParamsMemorySize();
		}


        wte_ = makePinVecf({config_.padded_vocab_size,config_.channels});//(Vp,C)
        wpe_ = makePinVecf({config_.max_seq_len,config_.channels});//(maxT,C)
        dwte_ = makePinVecf({config_.padded_vocab_size,config_.channels});//(Vp,C)
        dwpe_ = makePinVecf({config_.max_seq_len,config_.channels});//(maxT,C)

        params_bytes_ += wte_.size() * sizeof(float);
        params_bytes_ += wpe_.size() * sizeof(float);
        params_bytes_ += dwte_.size() * sizeof(float);
        params_bytes_ += dwpe_.size() * sizeof(float);

        lnf_ = makePinVecf({B,T,config_.channels});
        lnf_means_= makePinVecf({B,T});//(B,T)
        lnf_rstds_= makePinVecf({B,T});//(B,T)
        lnf_gamma_ = makePinVecf({config_.channels}) ;//(C)
        lnf_beta_  = makePinVecf({config_.channels}) ;//(C)

        params_bytes_ += lnf_.size() * sizeof(float);
        params_bytes_ += lnf_means_.size() * sizeof(float);
        params_bytes_ += lnf_rstds_.size() * sizeof(float);
        params_bytes_ += lnf_gamma_.size() * sizeof(float);
        params_bytes_ += lnf_beta_.size() * sizeof(float);

        residual_ = StdVec<PinVecf>(config_.num_layers,makePinVecf({B,T,config_.channels}));

        params_bytes_ += residual_.size() * sizeof(PinVecf);


        logits_ = makePinVecf({B,T,config_.padded_vocab_size});
        probs_ = makePinVecf({B,T,config_.padded_vocab_size});
        losses = makePinVecf({B,T});
        params_bytes_ += logits_.size() * sizeof(float);
        params_bytes_ += probs_.size() * sizeof(float);
        params_bytes_ += losses.size() * sizeof(float);


        dlnf_gamma_ = makePinVecf({config_.channels});//(C)
        dlnf_beta_  = makePinVecf({config_.channels});//(C)
        dlogits_ = makePinVecfZero({B,T,config_.padded_vocab_size});
        dlnf_ = makePinVecfZero({B,T,config_.channels});
        dresidual3_ = StdVec<PinVecf>(config_.num_layers,makePinVecfZero({B,T,config_.channels}));
        dencoded_ = makePinVecfZero({B,T,config_.channels});

        params_bytes_ += dlnf_gamma_.size() * sizeof(float);
        params_bytes_ += dlnf_beta_.size() * sizeof(float);
        params_bytes_ += dlogits_.size() * sizeof(float);
        params_bytes_ += dlnf_.size() * sizeof(float);
        params_bytes_ += dencoded_.size() * sizeof(float);

    }



	//params and grads data
	//will be used for updates
	{
		params_memory_.emplace_back(wte_.data(), wte_.size());
		grads_memory_.emplace_back(dwte_.data(), dwte_.size());

		params_memory_.emplace_back(wpe_.data(), wpe_.size());
		grads_memory_.emplace_back(dwpe_.data(), dwpe_.size());
		for (auto &layer : layers_) {
			params_memory_.emplace_back(layer.l_ln1_gamma.data(), layer.l_ln1_gamma.size());
			grads_memory_.emplace_back(layer.dl_ln1_gamma.data(), layer.dl_ln1_gamma.size());
			params_memory_.emplace_back(layer.l_ln1_beta.data(), layer.l_ln1_beta.size());
			grads_memory_.emplace_back(layer.dl_ln1_beta.data(), layer.dl_ln1_beta.size());

			params_memory_.emplace_back(layer.l_qkv_weight.data(), layer.l_qkv_weight.size());
			grads_memory_.emplace_back(layer.dl_qkv_weight.data(), layer.dl_qkv_weight.size());
			params_memory_.emplace_back(layer.l_qkv_bias.data(), layer.l_qkv_bias.size());
			grads_memory_.emplace_back(layer.dl_qkv_bias.data(), layer.dl_qkv_bias.size());

			params_memory_.emplace_back(layer.l_attproj_weight.data(), layer.l_attproj_weight.size());
			grads_memory_.emplace_back(layer.dl_attproj_weight.data(), layer.dl_attproj_weight.size());
			params_memory_.emplace_back(layer.l_attproj_bias.data(), layer.l_attproj_bias.size());
			grads_memory_.emplace_back(layer.dl_attproj_bias.data(), layer.dl_attproj_bias.size());

			params_memory_.emplace_back(layer.l_ln2_gamma.data(), layer.l_ln2_gamma.size());
			grads_memory_.emplace_back(layer.dl_ln2_gamma.data(), layer.dl_ln2_gamma.size());
			params_memory_.emplace_back(layer.l_ln2_beta.data(), layer.l_ln2_beta.size());
			grads_memory_.emplace_back(layer.dl_ln2_beta.data(), layer.dl_ln2_beta.size());

			params_memory_.emplace_back(layer.l_fch_weight.data(), layer.l_fch_weight.size());
			grads_memory_.emplace_back(layer.dl_fch_weight.data(), layer.dl_fch_weight.size());
			params_memory_.emplace_back(layer.l_fch_bias.data(), layer.l_fch_bias.size());
			grads_memory_.emplace_back(layer.dl_fch_bias.data(), layer.dl_fch_bias.size());

			params_memory_.emplace_back(layer.l_fcproj_weight.data(), layer.l_fcproj_weight.size());
			grads_memory_.emplace_back(layer.dl_fcproj_weight.data(), layer.dl_fcproj_weight.size());
			params_memory_.emplace_back(layer.l_fcproj_bias.data(), layer.l_fcproj_bias.size());
			grads_memory_.emplace_back(layer.dl_fcproj_bias.data(), layer.dl_fcproj_bias.size());
		}

		params_memory_.emplace_back(lnf_gamma_.data(), lnf_gamma_.size());
		grads_memory_.emplace_back(dlnf_gamma_.data(), dlnf_gamma_.size());
		params_memory_.emplace_back(lnf_beta_.data(), lnf_beta_.size());
		grads_memory_.emplace_back(dlnf_beta_.data(), dlnf_beta_.size());
	}
	//AdamW's m and v
	{
        optimizer_bytes_ = 0;
		size_t params_size = params_memory_.size();
		m_ = StdVec<PinVecf>(params_size);
		v_ = StdVec<PinVecf>(params_size);
		for (int i = 0; i < params_size; ++i) {
			m_[i] = makePinVecfZero({params_memory_[i].second});
			v_[i] = makePinVecfZero({params_memory_[i].second});
            optimizer_bytes_ += m_[i].size() * sizeof(float);
            optimizer_bytes_ += v_[i].size() * sizeof(float);
		}
        
	}

}

void GPT2::Forward(const StdVeci &inputs, const StdVeci &targets){
    assert(B_ * T_ == inputs.size());
    if(!targets.empty()){
        assert(B_ * T_ == targets.size());
    }

    this->inputs_.assign(inputs.begin(),inputs.end()) ;
    this->targets_.assign(targets.begin(),targets.end());

    auto L = config_.num_layers,B = B_,T = T_,C = config_.channels;
    auto NH = config_.num_heads,Vp = config_.padded_vocab_size,V = config_.vocab_size;
    auto MaxT = config_.max_seq_len;


    BatchEncoderForward(encoded_.data(), inputs_.data(),wte_.data(),wpe_.data(),B,T,C,Vp,MaxT);

    layers_.front().Forward(residual_.front(),encoded_);

    for(int l = 1 ; l < L; ++l){
        layers_[l].Forward(residual_[l],residual_[l-1]);
    }

    BatchLayerNormForward(lnf_.data(),lnf_means_.data(),lnf_rstds_.data(),residual_.back().data(),lnf_gamma_.data(),lnf_beta_.data(),B,T,C);

    BatchMatmulNTForward(logits_.data(), lnf_.data(),wte_.data(),nullptr, B,T,C,Vp);

    BatchSoftmaxForward(probs_.data(), logits_.data(), B, T, V,Vp);

    if(targets_.size() > 0 ) {
        BatchCrossEntropyForward(losses.data(),probs_.data(),targets_.data(),B,T,Vp);
        mean_loss = std::optional<float>(0.0f);
        mean_loss = std::accumulate(losses.begin(),losses.end(),mean_loss.value()) / (B*T);
    }else {
        mean_loss = std::nullopt;
    }
}



void GPT2::Backward(){
    auto L = config_.num_layers,B = B_,T = T_,C = config_.channels;
    auto NH = config_.num_heads,Vp = config_.padded_vocab_size,V = config_.vocab_size;
    auto MaxT = config_.max_seq_len;
    // std::cout << " ---backward--- " << std::endl;

    // std::cout << dlogits_.front() << std::endl;
    BatchCrossEntropySoftmaxBackward(dlogits_.data(), probs_.data(), targets_.data(),B,T,V,Vp,1.f/(B*T));
    
    // std::cout << probs_[0] << std::endl;
    // std::cout << targets_[0] << std::endl;

    // std::cout << dlogits_[0] << std::endl;

    // std::cout << dlnf_.front()  << std::endl;
    // std::cout << dwte_.front()  << std::endl;
    BatchMatmulNTBackward(dlnf_.data(),dwte_.data(),nullptr,dlogits_.data(),lnf_.data(),wte_.data(),B,T,C,Vp);
    // std::cout << dlnf_[0] << std::endl;

    // std::cout << dresidual3_.back().front()  << std::endl;
    // std::cout << dwte_.front()  << std::endl;
    BatchLayerNormBackward(dresidual3_.back().data(), dlnf_gamma_.data(), dlnf_beta_.data(), dlnf_.data(), residual_.back().data(), lnf_gamma_.data(), lnf_means_.data(),lnf_rstds_.data(),B,T,C);
    // std::cout << dresidual3_.back()[0] << std::endl;

    for(int l = L - 1; l > 0 ; --l){
        // std::cout << dresidual3_[l - 1].front()  << std::endl;
        layers_[l].Backward(dresidual3_[l - 1],dresidual3_[l],residual_[l-1]);
        // std::cout << dresidual3_[l-1][0] << std::endl;
    }

    // std::cout << dencoded_.front() << std::endl;
    layers_.front().Backward(dencoded_ ,dresidual3_.front(), encoded_);

    // std::cout << dwte_.front() << std::endl;
    // std::cout << dwpe_.front() << std::endl;
    BatchEncoderBackward(dwte_.data(),dwpe_.data(),dencoded_.data(),inputs_.data(),B,T,C,Vp,MaxT);
    // std::cout << dencoded_[0] << std::endl;

}


void GPT2::Update(float lr, float beta1, float beta2, float eps, float weight, int t) {

	// size_t params_size = params_memory_.size();
	// for (int i = 0; i < params_size; ++i) {
	// 	auto [data, size] = params_memory_[i];
	// 	auto [grads_data, grads_size] = grads_memory_[i];
	// 	assert(size == grads_size && size == m_[i].size());
    //     AdamW(data,grads_data,m_[i].data(),v_[i].data(),m_[i].size(),lr,beta1,beta1,eps,weight,t);
	// }
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


    std::cout << "Update Successfully!"  << std::endl;
}

void SetZero(PinVecf& v){
    v.assign(v.size(),0);
}

void SetZero(StdVec<PinVecf>& v){
    for(auto & vv : v){
        SetZero(vv);
    }
}
void ZeroGrad(Layer& l){
        SetZero(l.dl_residual2);    

        SetZero(l.dl_fcproj);       
        SetZero(l.dl_fcproj_weight );
        SetZero(l.dl_fcproj_bias);  

        SetZero(l.dl_fch_gelu);     

        SetZero(l.dl_fch);          
        SetZero(l.dl_fch_weight);   
        SetZero(l.dl_fch_bias);     

        SetZero(l.dl_ln2);          
        SetZero(l.dl_ln2_gamma);    
        SetZero(l.dl_ln2_beta);     



        SetZero(l.dl_attproj);      
        SetZero(l.dl_attproj_weight);
        SetZero(l.dl_attproj_bias); 

        SetZero(l.dl_atty);         

        SetZero(l.dl_qkv);          
        SetZero(l.dl_qkv_weight);   
        SetZero(l.dl_qkv_bias);     

        SetZero(l.dl_ln1);          
        SetZero(l.dl_ln1_gamma);    
        SetZero(l.dl_ln1_beta);     
}

void GPT2::ZeroGrad(){
    SetZero(dlogits_);
    SetZero(dlnf_);
    SetZero(dresidual3_);
    SetZero(dencoded_);
    SetZero(dwte_);
    SetZero(dwpe_);

    for(auto & layer : layers_){
        gpt2cuda::ZeroGrad(layer);
    }


    SetZero(dlnf_);
    SetZero(dlnf_gamma_);
    SetZero(dlnf_beta_);
}

int GPT2::Sample(int b,int t ,float coin, SampleMethod method){
    if(method == SampleMethod::Mult){
        float cdf = 0.f;
        float * probs_bt = probs_.data() + b*T_*config_.padded_vocab_size + t * config_.padded_vocab_size;
        for (int i = 0; i < config_.padded_vocab_size; ++i) {
            cdf += probs_bt[i];
            if (coin < cdf) {
                return i;
            }
        }
        return config_.padded_vocab_size - 1;
    }else {
        ERROR_PRINTLN("SampleMethod get a imposible value!");
        exit(-1);
    }
}

}
