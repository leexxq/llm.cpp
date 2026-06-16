#pragma once
#include "global.cuh"
#include <cassert>
#include <iostream>


void ReduceKernel(dim3 blocks,dim3 threads, float*dst,float const * src , int length,int stride);


namespace gpt2cuda {
    template <int threads = 256>
    void Reduce1D(float* dst , float const* src,int length,int stride){
            // dst.copy_from_host(dst);
            // d_bias calculate
            constexpr int warp_threads = 32;
            static_assert(threads % warp_threads == 0);
            constexpr int warp_count = threads / warp_threads;
            // constexpr int threads = warp_threads * warp_count;
            const int blocks = (stride+ warp_count - 1)/ warp_count;
            // ... d_bias ...    d_outputs be view as  [(B*T),Oc]
            // std::cout << "blocks:" << blocks << "threads:"<<threads << std::endl;
            ReduceKernel(blocks,threads,dst,src,length,stride);
            auto error = cudaGetLastError();
            if(error != cudaSuccess){
                std::cerr << cudaGetErrorString(error) << std::endl;
                exit(EXIT_FAILURE);
            }
            error = cudaDeviceSynchronize();
            if(error != cudaSuccess){
                std::cerr << cudaGetErrorString(error) << std::endl;
                exit(EXIT_FAILURE);
            }
    }

}