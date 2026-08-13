#pragma once

#include <filesystem>
#include <optional>
#include <utility>
#include "pinvector.cuh"
#include "layer.cuh"

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
        using Data_t = std::pair<float*, size_t>;

    private:
        GPT2Config config_;
        // the weights (parameters) of the model, and their sizes
        size_t num_parameters;

        StdVec<Data_t> params_memory_;
        StdVec<Data_t> grads_memory_;
        // gradients of the weights
        // buffers for the AdamW optimizer
        StdVec<PinVecf> m_;
        StdVec<PinVecf> v_;
        // the activations of the model, and their sizes
        size_t num_activations;
        // gradients of the activations
        // other run state configuration
        // size_t batch_size; // the batch size (B) of current forward pass
        // size_t seq_len; // the sequence length (T) of current forward pass
        PinVeci inputs_; // the input tokens for the current forward pass
        PinVeci targets_; // the target tokens for the current forward pass

        // Encoder encoder_;
        // LayerNorm layernormf_;
        // CrossEntropy loss_;

        StdVec<Layer> layers_;

        size_t B_;
        size_t T_;

        PinVecf wte_;//(Vp,C)
        PinVecf wpe_;//(maxT,C)

        PinVecf dwte_; //(Vp,C)
        PinVecf dwpe_; //(maxT,C)

        PinVecf encoded_;//(B,T,C)
        StdVec<PinVecf> residual_;//(L,B,T,C)
        PinVecf lnf_;//(B,T,C)
        
        PinVecf lnf_means_;//(B,T)
        PinVecf lnf_rstds_;//(B,T)

        PinVecf lnf_gamma_;//(C)
        PinVecf lnf_beta_;//(C)

        PinVecf dlnf_gamma_;//(C)
        PinVecf dlnf_beta_;//(C)
        

        PinVecf logits_;//(B,T,Vp)

        PinVecf dlogits_;//(B,T,Vp)
        PinVecf dlnf_;//(B,T,C)
        StdVec<PinVecf> dresidual3_;//(L,B,T,C)
        PinVecf dencoded_;//(B,T,C)

        

        std::size_t params_bytes_;
        std::size_t optimizer_bytes_;
        std::filesystem::path checkpoint_path_;
        PinVecf probs_;//(B,T,Vp)
    public:
        PinVecf losses; // (B,T)
        std::optional<float> mean_loss; // after a forward pass with targets, will be populated with the mean loss

    private:
        void Init(size_t B, size_t T);

    public:
        enum class SampleMethod{
            Mult,
        };
        GPT2() : config_{ 1024, 50257, 50304, 12, 12, 768 } {}
        GPT2(GPT2Config config, size_t B, size_t T) : config_{ config }, B_(B), T_(T) {}
        GPT2(const std::filesystem::path &path, size_t B, size_t T);
        GPT2(const GPT2 &gpt2) = delete;
        GPT2(const GPT2 &&gpt2) = delete;
        void Forward(const StdVeci &, const StdVeci &);
        void Forward(const  StdVeci &inputs) {
            StdVec<int> targets;
            Forward(inputs, targets);
        }
        void Backward();
        void ZeroGrad();
        void Update(float lr, float beta1, float beta2, float eps, float weight, int t);
        using VecBTC = StdVec<StdVec<PinVecf>>;
        int Sample(int b,int t ,float coin, SampleMethod method);
    };

}
