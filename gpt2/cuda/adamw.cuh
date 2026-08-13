#pragma once
#include "devvector.cuh"
namespace gpt2cuda {
    struct AdamWConfig{
    float lr;float beta1;float beta2;float eps;float weight_decay;int t;
    };
    void AdamW(float* data, float const * grad_data, float* m, float* v, std::size_t size, 
    float lr,float beta1,float beta2,float eps,float weight_decay,int t);
    void BatchAdamW(float** datas, float const ** grad_datas, float** m, float** v,std::size_t* sizes, std::size_t sizes_size,
    float lr,float beta1,float beta2,float eps,float weight_decay,int t);

    void AdamW(DevVecf& data , const DevVecf& grad_data , DevVecf& m ,DevVecf& v,
        std::size_t size, const AdamWConfig& config,cudaStream_t stream);
}