#include "adamw.cuh"
#include "cuda/global.cuh"
#include "cutlass/util/device_memory.h"
#include "log.h"
#include <cstdlib>
#include <iostream>

namespace gpt2cuda{
namespace kernel {

__global__ void AdamWKernel(
float* data ,float* grad_data, float *m ,float * v ,  std::size_t size, 
        float lr,float beta1,float beta2,float eps,float weight_decay,int t
){
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < size){
        float param = data[idx];
        float grad = grad_data[idx];

        float m_ = beta1 * m[idx] + (1.0f - beta1) * grad;
        float v_ = beta2 * v[idx] + (1.0f - beta2) * grad * grad;

        float m_hat = m_ / (1.0f - powf(beta1, t));
        float v_hat = v_ / (1.0f - powf(beta2, t));

        m[idx] = m_;
        v[idx] = v_;

        data[idx] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

template<int threads=256>
void AdamWCUDA(float* data ,float* grad_data, float *m ,float * v ,  std::size_t size, 
    float lr,float beta1,float beta2,float eps,float weight_decay,int t){

        AdamWKernel<<<(size + threads - 1) / threads,threads>>>(data, grad_data, m, v, size, lr, beta1, beta2, eps, weight_decay, t);
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

    kernel::AdamWCUDA(data_d.get(), grad_data_d.get(), m_d.get(), v_d.get(), size, lr, beta1, beta2, eps, weight_decay, t);

    data_d.copy_to_host(data);
    m_d.copy_to_host(m);
    v_d.copy_to_host(v);
}

void BatchAdamW(float** datas, float const ** grad_datas, float** m, float** v,std::size_t* sizes, std::size_t sizes_size,
float lr,float beta1,float beta2,float eps,float weight_decay,int t){
    // std::size_t datas_size = std::accumulate(sizes,sizes + sizes_size,0);

    std::size_t free_byte;
    std::size_t total_byte;

    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));

    std::size_t datas_size = 0;
    for(int i =0,j =0 ; j < sizes_size;){

        if((datas_size+sizes[i]) * sizeof(float) > free_byte){

            DAlloc datas_d(datas_size);
            DAlloc grads_d(datas_size);
            DAlloc m_d(datas_size);
            DAlloc v_d(datas_size);
            
            if(i == j){
                std::cerr << "BatchAdamW's data " << i << "out gpu memory!" << std::endl;
                exit(EXIT_FAILURE);
            }

            auto copy_host_lambda =[&](float* p)
            {
                for( int k = i;k < j ; ++k){
                    cutlass::device_memory::copy_to_device(p ,datas[i],sizes[i]);
                    p += sizes[i];
                }
            };

            auto copy_device_lambda =[&](float* p)
            {
                for( int k = i;k < j ; ++k){
                    cutlass::device_memory::copy_to_host(datas[i],p,sizes[i]);
                    p += sizes[i];
                }
            };

            copy_host_lambda(datas_d.get());
            copy_host_lambda(grads_d.get());
            copy_host_lambda(m_d.get());
            copy_host_lambda(v_d.get());

            kernel::AdamWCUDA(datas_d.get(),grads_d.get(),m_d.get(),v_d.get(),datas_size,lr,beta1,beta2,eps,weight_decay,t);

            copy_device_lambda(datas_d.get());
            copy_device_lambda(m_d.get());
            copy_device_lambda(v_d.get());

            datas_size = 0;
            i = j;
        }else {
            ++j;
        }
    }

    

}

}
