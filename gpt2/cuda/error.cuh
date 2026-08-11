#pragma once
#define CUDA_ERROR_HEAD() std::cerr << "[CUDA ERROR IN " << __FILE__ << ":" << __LINE__ << "]" 
#define CUDA_CHECK_LAST() do{auto error = cudaGetLastError();if(error != cudaSuccess){CUDA_ERROR_HEAD();std::cerr  <<cudaGetErrorString(error) << std::endl;exit(EXIT_FAILURE);}}while(false)
#define CUDA_CHECK(error) do{if(error != cudaSuccess){CUDA_ERROR_HEAD(); std::cerr<< cudaGetErrorString(error) << std::endl;exit(EXIT_FAILURE);}}while(false)