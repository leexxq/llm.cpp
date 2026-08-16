#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include "pinvector.cuh"
#include "devvector.cuh"
#include "layer.cuh"
#include<random>

namespace gpt2cuda {

    using size_t = std::size_t;
    struct GPT2Config {
    public:
        size_t max_seq_len; // max sequence length, e.g. 1024
        size_t vocab_size; // vocab size, e.g. 50257
        size_t padded_vocab_size; // padded to e.g. %128==0, 50304
        size_t num_layers; // number of layers, e.g. 12
        size_t num_heads; // number of heads in attention, e.g. 12
        size_t channels; // number of channels, e.g. 768
                        //
        void Print() const;
    };


    class GPT2 {
    public:
        enum class SampleMethod{
            Mult,
        };


    private:
        GPT2Config config_;
        // the weights (parameters) of the model, and their sizes
        size_t num_parameters;
        

        StdVec<DevVecf*> params_memory_;
        StdVec<DevVecf*> grads_memory_;
        // gradients of the weights
        // buffers for the AdamW optimizer
        StdVec<DevVecf> m_;
        StdVec<DevVecf> v_;
        // the activations of the model, and their sizes
        size_t num_activations;
        // gradients of the activations
        // other run state configuration
        // size_t batch_size; // the batch size (B) of current forward pass
        // size_t seq_len; // the sequence length (T) of current forward pass
        DevVeci inputs_; // the input tokens for the current forward pass
        DevVeci targets_; // the target tokens for the current forward pass

        // Encoder encoder_;
        // LayerNorm layernormf_;
        // CrossEntropy loss_;

        StdVec<Layer> layers_;

        size_t B_;
        size_t T_;

        DevVecf wte_;//(Vp,C)
        DevVecf wpe_;//(maxT,C)

        DevVecf dwte_; //(Vp,C)
        DevVecf dwpe_; //(maxT,C)

        DevVecf encoded_;//(B,T,C)
        StdVec<DevVecf> residual_;//(L,B,T,C)
        DevVecf lnf_;//(B,T,C)
        
        DevVecf lnf_means_;//(B,T)
        DevVecf lnf_rstds_;//(B,T)

        DevVecf lnf_gamma_;//(C)
        DevVecf lnf_beta_;//(C)

        DevVecf dlnf_gamma_;//(C)
        DevVecf dlnf_beta_;//(C)
        

        DevVecf logits_;//(B,T,Vp)

        DevVecf dlogits_;//(B,T,Vp)
        DevVecf dlnf_;//(B,T,C)
        StdVec<DevVecf> dresidual3_;//(L,B,T,C)
        DevVecf dencoded_;//(B,T,C)

        

        std::size_t params_bytes_;
        std::size_t optimizer_bytes_;
        std::filesystem::path checkpoint_path_;

    public:
        DevVecf probs_;//(B,T,Vp)
        DevVecf losses; // (B,T)
        float mean_loss = -1.f; 

    private:
        void CUDART_CB static on_probs_ready(void *userData);
        void Init(size_t B, size_t T);
    public:
        GPT2() : config_{ 1024, 50257, 50304, 12, 12, 768 }{}
        GPT2(GPT2Config config, size_t B, size_t T) : config_{ config }, B_(B), T_(T){Init(B, T);}
        GPT2(const std::filesystem::path &path, size_t B, size_t T);
        GPT2(const GPT2 &gpt2) = delete;
        GPT2(const GPT2 &&gpt2) = delete;
        void Forward(const StdVeci &, const StdVeci &,cudaStream_t stream);
        void Forward(const  StdVeci &inputs,cudaStream_t stream) {
            StdVec<int> targets;
            Forward(inputs, targets,stream);
        }
        void GetProbs(PinVecf& host,cudaStream_t stream){
            probs_.to(host,stream);
        }
        size_t GetProbsSize() const{
            return B_*T_*config_.padded_vocab_size;
        }
        float GetLoss(cudaStream_t stream) ;
        void ZeroLoss(cudaStream_t stream) ;
        


        void Backward(cudaStream_t stream);
        void ZeroGrad(cudaStream_t stream);
        void Update(float lr, float beta1, float beta2, float eps, float weight, int t,cudaStream_t stream);
    };

}
