#pragma once

#include <cstddef>
#include <filesystem>
#include <ios>
#include <memory>
#include <stdexcept>
#include "pinvector.cuh"
#include "devvector.cuh"
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
        enum class SampleMethod{
            Mult,
        };
        struct Stream{
            private:
                cudaStream_t cu_stream;
                bool is_defalut_stream;
                bool is_only_refer;
                DevVeci inputs;
                DevVeci targets;
                friend GPT2;
                Stream(std::size_t B , std::size_t T,cudaStream_t cu_stream,bool default_stream,bool only_refer)
                : cu_stream(cu_stream),inputs(B*T),is_defalut_stream(default_stream),is_only_refer(only_refer) {
                    if(!is_only_refer){
                        targets=makeDevVeci(B*T);
                    }
                }
                
            public: 
                Stream() = delete;
                Stream(const Stream& other) = delete;
                Stream& operator=(const Stream& other) = delete;
                Stream(Stream&& other) = delete;
                Stream& operator=(Stream&& other) = delete;
                void SetOnlyRefer(bool is_only_refer) {
                    this->is_only_refer =is_only_refer;
                }
                bool GetOnlyRefer() const{
                    return this->is_only_refer;
                }
                cudaStream_t GetStream() const {
                    return cu_stream;
                }
                ~Stream(){
                    cudaStreamDestroy(cu_stream);
                }
        };
        enum StreamOption{
            NOP = 1L << 0,
            DefaultStream = 1L << 1,
            OnlyRefer = 1L << 2,
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

    private:
        void Init(size_t B, size_t T);
    public:
        GPT2() : config_{ 1024, 50257, 50304, 12, 12, 768 }{}
        GPT2(GPT2Config config, size_t B, size_t T) : config_{ config }, B_(B), T_(T){Init(B, T);}
        GPT2(const std::filesystem::path &path, size_t B, size_t T);
        GPT2(const GPT2 &gpt2) = delete;
        GPT2(const GPT2 &&gpt2) = delete;

        void Forward(Stream& stream);
        void Forward(DevVecf& losses , Stream& stream);


        [[__nodiscard__("stream will used train later")]]Stream CreateStream(StreamOption options = StreamOption::NOP) const {
            cudaStream_t cu_stream;

            bool is_default_stream = false;
            if((options & StreamOption::DefaultStream) == 0){
                CUDA_CHECK(cudaStreamCreate(&cu_stream));
                cu_stream = cu_stream;
            }else{
                cu_stream = cudaStreamDefault;
            }
            bool is_only_refer = false;
            if((options & StreamOption::OnlyRefer) > 0){
                is_only_refer = true;
            }
            return {B_,T_,cu_stream,is_default_stream,is_only_refer};
        };
        void SetStreamOnlyRefer(Stream& s)const{
            s.targets = DevVeci();
        }
        
        void SetTrainData( Stream& stream,const PinVeci& inputs, const PinVeci& targets=PinVeci());

        

        void GetProbs(PinVecf& host,const Stream& stream){
            probs_.to(host,stream.GetStream());
        }
        size_t GetProbsSize() const{
            return B_*T_*config_.padded_vocab_size;
        }

        float GetLossSync(const Stream& stream);
        void ZeroLoss(Stream& stream);
        


        void Backward(Stream& stream);
        void ZeroGrad(Stream& stream);
        void Update(float lr, float beta1, float beta2, float eps, float weight, int t,Stream &stream);
    };

}
