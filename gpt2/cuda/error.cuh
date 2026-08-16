#pragma once
#include <cutlass/util/exceptions.h>
#define CUDA_ERROR_HEAD() std::cerr << "[CUDA ERROR IN " << __FILE__ << ":" << __LINE__ << "]" 
#define CUDA_CHECK_LAST() do{auto error = cudaGetLastError();if(error != cudaSuccess){ throw cutlass::cuda_exception(cudaGetErrorString(error),error);}}while(false)
#define CUDA_CHECK(error) do{if(error != cudaSuccess){ throw cutlass::cuda_exception(cudaGetErrorString(error),error);}}while(false)