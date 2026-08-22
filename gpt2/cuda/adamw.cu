#include "adamw.cuh"
#include "error.cuh"
#include "cutlass/util/device_memory.h"
#include <cstdlib>
namespace gpt2cuda{
namespace kernel {

__global__ void AdamWKernel(
float* data ,float const* grad_data, float *m ,float * v ,  std::size_t size, 
        float lr,float beta1,float beta2,float eps,float weight_decay,float rpow2_beta1_t,float rpow2_beta2_t
){
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < size){
        float param = data[idx];
        float grad = grad_data[idx];

        float m_ = beta1 * m[idx] + (1.0f - beta1) * grad;
        float v_ = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        

        float m_hat = m_ * rpow2_beta1_t;
        float v_hat = v_ * rpow2_beta2_t;

        m[idx] = m_;
        v[idx] = v_;

        

        data[idx] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
        // data[idx] -= lr * (m_hat * __frsqrt_rn(v_hat + eps) + weight_decay * param);
    }
}

template<int threads=256>
void AdamWCUDA(float* data ,float const* grad_data, float *m ,float * v ,  std::size_t size, 
    float lr,float beta1,float beta2,float eps,float weight_decay,int t,cudaStream_t stream){
        auto beta1_t = 1.0f/ ( 1.0f - std::pow(beta1,t));
        auto beta2_t = 1.0f / (1.0f - std::pow(beta2,t));
        AdamWKernel<<<(size + threads - 1) / threads,threads,0,stream>>>(data, grad_data, m, v, size, lr, beta1, beta2, eps, weight_decay, beta1_t,beta2_t);
        CUDA_CHECK_LAST();
}

}


using DAlloc = cutlass::device_memory::allocation<float>;
void AdamW(float* data, float const* grad_data, float* m, float* v, std::size_t size, 
        float lr,float beta1,float beta2,float eps,float weight_decay,int t){
    DAlloc data_d(size);
    DAlloc grad_data_d(size);
    DAlloc m_d(size);
    DAlloc v_d(size);
    data_d.copy_from_host(data);
    grad_data_d.copy_from_host(grad_data);
    m_d.copy_from_host(m);
    v_d.copy_from_host(v);

    kernel::AdamWCUDA(data_d.get(), grad_data_d.get(), m_d.get(), v_d.get(), size, lr, beta1, beta2, eps, weight_decay, t,0);

    data_d.copy_to_host(data);
    m_d.copy_to_host(m);
    v_d.copy_to_host(v);
}

void AdamW(DevVecf& data , const DevVecf& grad_data , DevVecf& m ,DevVecf& v,
        std::size_t size, const AdamWConfig& config,cudaStream_t stream){
    kernel::AdamWCUDA(data.data(), grad_data.data(), m.data(), v.data(), size, config.lr,config.beta1,config.beta2,config.eps,config.weight_decay,config.t,stream);
}

}
