#include "cuda/attention.cuh"
#include "attention.h"
#include "global.h"
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <chrono>
#include <cstddef>
#include <numeric>
#include <random>

using MatfRow = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using MatiRow = Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;


TEST(CudaAttention,forward1){
	constexpr size_t B = 4;
	constexpr size_t T = 64;
    constexpr size_t C = 768;
    constexpr size_t C3 = 3 * C;
    constexpr size_t NH = 24 ;

    auto att =  Attention(B,T,C,NH);

    VecBTC inputs(B);
	StdVec<float> inputs_vec(B*T*C3);
	for(int i =0 ; i < B ; ++i){
		inputs[i] = Matf::Random(T,C3);
        Eigen::Map<MatfRow> (inputs_vec.data() + i * T*C3 , T,C3)  = inputs[i];
	}


	VecBTC outputs_res ;
	// {
	// 	auto start = std::chrono::high_resolution_clock::now();
	// 	outputs_res = att.Forward(inputs);
	// 	auto end = std::chrono::high_resolution_clock::now();
	// 	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	// 	std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	// }

    StdVec<float> outputs_vec(B*T*C);

	{
		auto start = std::chrono::high_resolution_clock::now();
        gpt2cuda::BatchAttentionForward(outputs_vec.data(), inputs_vec.data(), B, T, C3,NH);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

    


}