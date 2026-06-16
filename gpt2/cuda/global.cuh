#pragma  once
#define CUDA_ERROR_HEAD() std::cerr << "[CUDA ERROR IN " << __FILE__ << ":" << __LINE__ << "]" 
#define CUDA_CHECK_LAST() do{auto error = cudaGetLastError();if(error != cudaSuccess){CUDA_ERROR_HEAD();std::cerr  <<cudaGetErrorString(error) << std::endl;exit(EXIT_FAILURE);}}while(false)
#define CUDA_CHECK(error) do{if(error != cudaSuccess){CUDA_ERROR_HEAD(); std::cerr<< cudaGetErrorString(error) << std::endl;exit(EXIT_FAILURE);}}while(false)

#include <vector>
namespace gpt2cuda {

    using StdVecf = std::vector<float>;
    inline StdVecf makeZero(int B,int T,int C){
        return StdVecf(B*T*C,0);
    }
    inline StdVecf makeVecf(int B,int T,int C){
        return StdVecf(B*T*C);
    }
    inline StdVecf makeVecf(int B,int T){
        return StdVecf(B*T);
    }

    inline StdVecf makeVecf(int B){
        return StdVecf(B);
    }

}



