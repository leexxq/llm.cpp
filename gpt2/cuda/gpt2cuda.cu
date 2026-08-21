#include "cuda/devvector.cuh"
#include "cuda/pinvector.cuh"
#include "cuda/softmax.cuh"
#include "layernorm.cuh"
#include "log.h"
#include "matmul.cuh"
#include "cross_entropy.cuh"
#include "encoder.cuh"
#include "gpt2cuda.h"
#include "adamw.h"
#include "adamw.cuh"
#include "nvtx3/nvtx3.hpp"
#include <cassert>
#include <iostream>
#include <numeric>


#include <fstream>
#include <stdexcept>
namespace gpt2cuda{

void GPT2Config::Print() const {
    std::cout << "max_seq_len:" << max_seq_len<<std::endl;
    std::cout << "vocab_size:" << vocab_size<<std::endl;
    std::cout << "padded_vocab_size:" <<padded_vocab_size<<std::endl;
    std::cout << "num_layers:" << num_layers<<std::endl;
    std::cout << "num_heads:" << num_heads<<std::endl;
    std::cout << "channels:" << channels<<std::endl;
}


GPT2::GPT2(const std::filesystem::path &path, size_t B, size_t T) : checkpoint_path_(path), B_(B), T_(T){
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
        PinVecf wte_buffer(C * Vp);
        f.read(reinterpret_cast<byte>(wte_buffer.data()),sizeof(float)*C * Vp);
        wte_ = wte_buffer;
        std::cout << "load success wte"  << std::endl;
        //wte(MaxT,C)
        PinVecf wpe_buffer(C * maxT);
        f.read(reinterpret_cast<byte>(wpe_buffer.data()),sizeof(float)*C * maxT);
        wpe_ = wpe_buffer;
        std::cout << "load success wpe"  << std::endl;

        for(int i =0 ; i < L; ++i){
            PinVecf buffer(C);
            f.read(reinterpret_cast<byte>(buffer.data()),sizeof(float)*C );
            layers_[i].l_ln1_gamma = buffer;
        }
        for(int i =0 ; i < L; ++i){
            PinVecf buffer(C);
            f.read(reinterpret_cast<byte>(buffer.data()),sizeof(float)*C );
            layers_[i].l_ln1_beta = buffer;
        }

        for(int i = 0 ; i < L ; ++i){
            PinVecf buffer(C*3*C);
            f.read(reinterpret_cast<byte>(buffer.data()),sizeof(float)*C* 3 *C );
            layers_[i].l_qkv_weight = buffer;
        }
        for(int i = 0 ; i < L ; ++i){
            PinVecf buffer(3*C);
            f.read(reinterpret_cast<byte>(buffer.data()),sizeof(float) * 3*C );
            layers_[i].l_qkv_bias = buffer;
        }
        for(int i = 0 ; i < L ; ++i){
            PinVecf buffer(C*C);
            f.read(reinterpret_cast<byte>(buffer.data()),sizeof(float) * C*C );
            layers_[i].l_attproj_weight = buffer;
        }
        for(int i = 0 ; i < L ; ++i){
            PinVecf buffer(C);
            f.read(reinterpret_cast<byte>(buffer.data()),sizeof(float) * C );
            layers_[i].l_attproj_bias = buffer;
        }
        for(int i = 0 ; i < L ; ++i){
            PinVecf buffer(C);
            f.read(reinterpret_cast<byte>(buffer.data()),sizeof(float) * C );
            layers_[i].l_ln2_gamma = buffer;
        }
        for(int i = 0 ; i < L ; ++i){
            PinVecf buffer(C);
            f.read(reinterpret_cast<byte>(buffer.data()),sizeof(float) * C );
            layers_[i].l_ln2_beta = buffer;
        }
        for(int i = 0 ; i < L ; ++i){
            PinVecf buffer(4*C*C);
            f.read(reinterpret_cast<byte>(buffer.data()),sizeof(float) * C * 4*C );
            layers_[i].l_fch_weight = buffer;
        }
        for(int i = 0 ; i < L ; ++i){
            PinVecf buffer(4*C);
            f.read(reinterpret_cast<byte>(buffer.data()),sizeof(float) *4*C );
            layers_[i].l_fch_bias = buffer;
        }
        for(int i = 0 ; i < L ; ++i){
            PinVecf buffer(C*4*C);
            f.read(reinterpret_cast<byte>(buffer.data()),sizeof(float) *C*4*C );
            layers_[i].l_fcproj_weight = buffer;
        }
        for(int i = 0 ; i < L ; ++i){
            PinVecf buffer(C);
            f.read(reinterpret_cast<byte>(buffer.data()),sizeof(float) *C);
            layers_[i].l_fcproj_bias = buffer;
        }

        {
            PinVecf buffer(C);
            f.read(reinterpret_cast<byte>(buffer.data()),sizeof(float) *C);
            lnf_gamma_ = buffer;
        }
        {
            PinVecf buffer(C);
            f.read(reinterpret_cast<byte>(buffer.data()),sizeof(float) *C);
            lnf_beta_ = buffer;
        }

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


    {
        //encoder init

        encoded_ = makeDevVecf(B*T*config_.channels);
        params_bytes_ += encoded_.size() * sizeof(float);

        //layers init
		for (int l = 0; l < config_.num_layers; ++l) {
			layers_.emplace_back(B, T, config_.channels, config_.vocab_size, config_.num_heads);
            params_bytes_ += layers_.back().GetParamsMemorySize();
		}


        wte_ = makeDevVecf(config_.padded_vocab_size*config_.channels);//(Vp,C)
        wpe_ = makeDevVecf(config_.max_seq_len*config_.channels);//(maxT,C)
        dwte_ = makeDevVecf(config_.padded_vocab_size*config_.channels);//(Vp,C)
        dwpe_ = makeDevVecf(config_.max_seq_len*config_.channels);//(maxT,C)

        params_bytes_ += wte_.size() * sizeof(float);
        params_bytes_ += wpe_.size() * sizeof(float);
        params_bytes_ += dwte_.size() * sizeof(float);
        params_bytes_ += dwpe_.size() * sizeof(float);

        lnf_ = makeDevVecf(B*T*config_.channels);
        lnf_means_= makeDevVecf(B*T);//(B,T)
        lnf_rstds_= makeDevVecf(B*T);//(B,T)
        lnf_gamma_ = makeDevVecf(config_.channels) ;//(C)
        lnf_beta_  = makeDevVecf(config_.channels) ;//(C)

        params_bytes_ += lnf_.size() * sizeof(float);
        params_bytes_ += lnf_means_.size() * sizeof(float);
        params_bytes_ += lnf_rstds_.size() * sizeof(float);
        params_bytes_ += lnf_gamma_.size() * sizeof(float);
        params_bytes_ += lnf_beta_.size() * sizeof(float);

        residual_ = StdVec<DevVecf>(config_.num_layers,makeDevVecf(B*T*config_.channels));

        params_bytes_ += residual_.size() * sizeof(DevVecf);



        logits_ = makeDevVecf(B*T*config_.padded_vocab_size);
        probs_ = makeDevVecf(B*T*config_.padded_vocab_size);
        losses = makeDevVecfZero(B*T);
        params_bytes_ += logits_.size() * sizeof(float);
        params_bytes_ += probs_.size() * sizeof(float);
        params_bytes_ += losses.size() * sizeof(float);


        dlnf_gamma_ = makeDevVecf(config_.channels);//(C)
        dlnf_beta_  = makeDevVecf(config_.channels);//(C)
        dlogits_ = makeDevVecfZero(B*T*config_.padded_vocab_size);
        dlnf_ = makeDevVecfZero(B*T*config_.channels);
        dresidual3_ = StdVec<DevVecf>(config_.num_layers,makeDevVecfZero(B*T*config_.channels));
        dencoded_ = makeDevVecfZero(B*T*config_.channels);

        params_bytes_ += dlnf_gamma_.size() * sizeof(float);
        params_bytes_ += dlnf_beta_.size() * sizeof(float);
        params_bytes_ += dlogits_.size() * sizeof(float);
        params_bytes_ += dlnf_.size() * sizeof(float);
        params_bytes_ += dencoded_.size() * sizeof(float);

    }



	//params and grads data
	//will be used for updates
	{
		params_memory_.emplace_back(&wte_);
		grads_memory_.emplace_back(&dwte_);

		params_memory_.emplace_back(&wpe_);
		grads_memory_.emplace_back(&dwpe_);
		for (auto &layer : layers_) {
			params_memory_.emplace_back(&layer.l_ln1_gamma);
			grads_memory_.emplace_back(&layer.dl_ln1_gamma);
			params_memory_.emplace_back(&layer.l_ln1_beta);
			grads_memory_.emplace_back(&layer.dl_ln1_beta);

			params_memory_.emplace_back(&layer.l_qkv_weight);
			grads_memory_.emplace_back(&layer.dl_qkv_weight);
			params_memory_.emplace_back(&layer.l_qkv_bias);
			grads_memory_.emplace_back(&layer.dl_qkv_bias);

			params_memory_.emplace_back(&layer.l_attproj_weight);
			grads_memory_.emplace_back(&layer.dl_attproj_weight);
			params_memory_.emplace_back(&layer.l_attproj_bias);
			grads_memory_.emplace_back(&layer.dl_attproj_bias);

			params_memory_.emplace_back(&layer.l_ln2_gamma);
			grads_memory_.emplace_back(&layer.dl_ln2_gamma);
			params_memory_.emplace_back(&layer.l_ln2_beta);
			grads_memory_.emplace_back(&layer.dl_ln2_beta);

			params_memory_.emplace_back(&layer.l_fch_weight);
			grads_memory_.emplace_back(&layer.dl_fch_weight);
			params_memory_.emplace_back(&layer.l_fch_bias);
			grads_memory_.emplace_back(&layer.dl_fch_bias);

			params_memory_.emplace_back(&layer.l_fcproj_weight);
			grads_memory_.emplace_back(&layer.dl_fcproj_weight);
			params_memory_.emplace_back(&layer.l_fcproj_bias);
			grads_memory_.emplace_back(&layer.dl_fcproj_bias);
		}
        
		params_memory_.emplace_back(&lnf_gamma_);
		grads_memory_.emplace_back(&dlnf_gamma_);
		params_memory_.emplace_back(&lnf_beta_);
		grads_memory_.emplace_back(&dlnf_beta_);
	}
	//AdamW's m and v
	{
        optimizer_bytes_ = 0;
		size_t params_size = params_memory_.size();
		m_ = StdVec<DevVecf>(params_size);
		v_ = StdVec<DevVecf>(params_size);
		for (int i = 0; i < params_size; ++i) {
			m_[i] = makeDevVecfZero(params_memory_[i]->size());
			v_[i] = makeDevVecfZero(params_memory_[i]->size());
            optimizer_bytes_ += m_[i].size() * sizeof(float);
            optimizer_bytes_ += v_[i].size() * sizeof(float);
		}
        
	}


}
void GPT2::Forward(Stream& s){
    Forward(this->losses,s);
}


void GPT2::Forward(DevVecf& losses,Stream& s){
    auto L = config_.num_layers,B = B_,T = T_,C = config_.channels;
    auto NH = config_.num_heads,Vp = config_.padded_vocab_size,V = config_.vocab_size;
    auto MaxT = config_.max_seq_len;

    auto stream = s.GetStream();
    auto& inputs = s.inputs;

    BatchEncoderForward(encoded_, inputs,wte_,wpe_,B,T,C,Vp,MaxT,stream);

    layers_.front().Forward(residual_.front(),encoded_,stream);

    for(int l = 1 ; l < L; ++l){
        layers_[l].Forward(residual_[l],residual_[l-1],stream);
    }

    BatchLayerNormForward(lnf_,lnf_means_,lnf_rstds_,residual_.back(),lnf_gamma_,lnf_beta_,B,T,C,stream);

    BatchMatmulNTForward(logits_, lnf_,wte_,DevVecf(), B,T,C,Vp,stream);

    BatchSoftmaxForward(probs_, logits_, B, T, V,Vp,stream);

    if(!s.is_only_refer) {
        BatchCrossEntropyForward(losses,probs_,s.targets,B,T,Vp,stream);
    }
}





void GPT2::SetTrainData(Stream& s,const PinVeci& inputs, const PinVeci& targets){
    auto stream = s.GetStream();
    if(inputs.size() != B_*T_){
        throw std::invalid_argument("invalid inputs size");
    }
    s.inputs.from(inputs.data(),stream);
    if(s.is_only_refer){
        if(targets.size() > 0){
            throw std::runtime_error("try set target to a only refer stream");
        }
    }else{
        if(targets.size() != B_*T_){
            throw std::invalid_argument("invalid targets size");
        }
        s.targets.from(targets.data(),stream);
    }
    
}


float GPT2::GetLossSync(const Stream& s){
    auto stream = s.GetStream();
    PinVecf losses_h(losses.size());
    losses.to(losses_h,stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float mean_loss = std::accumulate(losses_h.begin(),losses_h.end(),0.f) / (B_*T_);
    return mean_loss;
}


void GPT2::ZeroLoss(Stream& s){
    auto stream = s.GetStream();
    losses.zero(stream);
}

// #define quick_debug_print(v,stream) \
// {\
//     PinVecf v##_h(v.size());\
//     v.to(v##_h,stream);\
//     cudaStreamSynchronize(stream);\
//     std::cout << #v": " << std::endl;\
//     int i =0;\
//     for(auto x : v##_h){\
//         if( ++i > 5) break;\
//         std::cout << x << " " ;\
//     }\
//     std::cout << std::endl;\
// }
#define quick_debug_print(v,stream)


void GPT2::Backward(Stream& s){

    auto stream = s.GetStream();
    auto& inputs = s.inputs;
    auto& targets = s.targets;

    auto L = config_.num_layers,B = B_,T = T_,C = config_.channels;
    auto NH = config_.num_heads,Vp = config_.padded_vocab_size,V = config_.vocab_size;
    auto MaxT = config_.max_seq_len;

    BatchCrossEntropySoftmaxBackward(dlogits_, probs_, targets,B,T,V,Vp,1.f/(B*T),stream);
    

    DevVecf dummy;
    BatchMatmulNTBackward(dlnf_,dwte_,dummy,dlogits_,lnf_,wte_,B,T,C,Vp,stream);

    BatchLayerNormBackward(dresidual3_.back(), dlnf_gamma_, dlnf_beta_, dlnf_, residual_.back(), lnf_gamma_, lnf_means_,lnf_rstds_,B,T,C,stream);

    for(int l = L - 1; l > 0 ; --l){
        layers_[l].Backward(dresidual3_[l - 1],dresidual3_[l],residual_[l-1],stream);
    }

    layers_.front().Backward(dencoded_ ,dresidual3_.front(), encoded_,stream);
    quick_debug_print(dencoded_,stream);

    BatchEncoderBackward(dwte_,dwpe_,dencoded_,inputs,B,T,C,Vp,MaxT,stream);
    quick_debug_print(dwte_,stream);

    
    
}

void GPT2::Update(float lr, float beta1, float beta2, float eps, float weight, int t,Stream& s) {

    auto stream = s.GetStream();
	size_t params_size = params_memory_.size();

	AdamWConfig adamw_params{
		lr,
		beta1,
		beta2,
		eps,
		weight,
		t
	};

	for (int i = 0; i < params_size; ++i) {
		auto& data = *params_memory_[i];
		auto& grad_data = *grads_memory_[i];
		auto size = data.size();
		auto grads_size = grad_data.size();
		assert(size == grads_size && size == m_[i].size());
        AdamW(data,grad_data,m_[i],v_[i],size,adamw_params,stream);
	}
	// AdamWParams adamw_params{
	// 	lr,
	// 	beta1,
	// 	beta2,
	// 	eps,
	// 	weight,
	// 	t
	// };

	// size_t params_size = params_memory_.size();
	// for (int i = 0; i < params_size; ++i) {
	// 	auto [data, size] = params_memory_[i];
	// 	auto [grads_data, grads_size] = grads_memory_[i];
	// 	assert(size == grads_size && size == m_[i].size());
	// 	AdamW(data, grads_data, m_[i].data(), v_[i].data(), m_[i].size(), adamw_params);
	// }

}

void SetZero(PinVecf& v){
    v.assign(v.size(),0);
}
void SetZero(DevVecf& v,cudaStream_t stream){
    v.zero(stream);
}

void SetZero(StdVec<PinVecf>& v){
    for(auto & vv : v){
        SetZero(vv);
    }
}

void SetZero(StdVec<DevVecf>& v,cudaStream_t stream){
    for(auto & vv : v){
        SetZero(vv,stream);
    }
}
void ZeroGrad(Layer& l,cudaStream_t stream){
        SetZero(l.dl_residual2,stream);    

        SetZero(l.dl_fcproj,stream);       
        SetZero(l.dl_fcproj_weight ,stream);
        SetZero(l.dl_fcproj_bias,stream);  

        SetZero(l.dl_fch_gelu,stream);     

        SetZero(l.dl_fch,stream);          
        SetZero(l.dl_fch_weight,stream);   
        SetZero(l.dl_fch_bias,stream);     

        SetZero(l.dl_ln2,stream);          
        SetZero(l.dl_ln2_gamma,stream);    
        SetZero(l.dl_ln2_beta,stream);     



        SetZero(l.dl_attproj,stream);      
        SetZero(l.dl_attproj_weight,stream);
        SetZero(l.dl_attproj_bias,stream); 

        SetZero(l.dl_atty,stream);         

        SetZero(l.dl_qkv,stream);          
        SetZero(l.dl_qkv_weight,stream);   
        SetZero(l.dl_qkv_bias,stream);     

        SetZero(l.dl_ln1,stream);          
        SetZero(l.dl_ln1_gamma,stream);    
        SetZero(l.dl_ln1_beta,stream);     
}

void GPT2::ZeroGrad(Stream& s){
    auto stream = s.GetStream();
    SetZero(dlogits_,stream);
    SetZero(dlnf_,stream);
    SetZero(dresidual3_,stream);
    SetZero(dencoded_,stream);
    SetZero(dwte_,stream);
    SetZero(dwpe_,stream);

    for(auto & layer : layers_){
        gpt2cuda::ZeroGrad(layer,stream);
    }


    SetZero(dlnf_,stream);
    SetZero(dlnf_gamma_,stream);
    SetZero(dlnf_beta_,stream);
}




    


}
