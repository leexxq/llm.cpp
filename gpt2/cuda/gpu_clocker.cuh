
#pragma once

#include "cuda/error.cuh"
#include <cuda_runtime.h>

struct GPUClock
{
    GPUClock() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~GPUClock() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() {
        cudaEventRecord(start_);
        this->stream = cudaStreamDefault;
    }
    void start(cudaStream_t stream) {
        cudaEventRecord(start_,stream);
        this->stream = stream;
    }

    float milliseconds() {
        CUDA_CHECK(cudaEventRecord(stop_,stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float time;
        CUDA_CHECK(cudaEventElapsedTime(&time, start_, stop_));
        return time;
    }

    float seconds() {
        return milliseconds() * float(1e-3);
    }

private:
    cudaEvent_t start_, stop_;
    cudaStream_t stream = cudaStreamDefault;
};