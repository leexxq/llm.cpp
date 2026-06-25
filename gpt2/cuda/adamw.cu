#include "adamw.cuh"
#include <cstddef>
#include "cuda/global.cuh"
#include "cutlass/util/device_memory.h"

namespace gpt2cuda{
namespace kernel {

__global__ void AdamWKernel(
float* data ,float* grad_data, float *m ,float * v ,  size_t size, 
        float lr,float beta1,float beta2,float eps,float weight_decay,int t
){
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < size){
        float param = data[idx];
        float grad = grad_data[idx];

        float m_ = beta1 * m[idx] + (1.0f - beta1) * grad;
        float v_ = beta2 * v[idx] + (1.0f - beta2) * grad * grad;

        float m_hat = m_ / (1.0f - std::pow(beta1, t));
        float v_hat = v_ / (1.0f - std::pow(beta2, t));

        m[idx] = m_;
        v[idx] = v_;

        data[idx] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * param);
    }
}

template<int threads=256>
void AdamWCuda(float* data ,float* grad_data, float *m ,float * v ,  size_t size, 
    float lr,float beta1,float beta2,float eps,float weight_decay,int t){

        AdamWKernel<<<(size + threads - 1) / threads,threads>>>(data, grad_data, m, v, size, lr, beta1, beta2, eps, weight_decay, t);
        CUDA_CHECK_LAST();

}

}


using DAlloc = cutlass::device_memory::allocation<float>;
void AdamW(float* data, float * grad_data, float* m, float* v, size_t size, 
        float lr,float beta1,float beta2,float eps,float weight_decay,int t){
    DAlloc data_d(size);
    DAlloc grad_data_d(size);
    DAlloc m_d(size);
    DAlloc v_d(size);
    data_d.copy_from_host(data);
    grad_data_d.copy_from_host(grad_data);
    m_d.copy_from_host(m);
    v_d.copy_from_host(v);

    kernel::AdamWCuda(data_d.get(), grad_data_d.get(), m_d.get(), v_d.get(), size, lr, beta1, beta2, eps, weight_decay, t);

    data_d.copy_to_host(data);
    m_d.copy_to_host(m);
    v_d.copy_to_host(v);
}

}
