#include "cuda/softmax.cuh"
#include "softmax.h"
#include "global.h"
#include <gtest/gtest.h>
#include <chrono>
#include <cstddef>

using MatfRow = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;


TEST(CudaSoftmax,no_padding){

    //too large cpu will slow
	constexpr size_t B = 4;
	constexpr size_t T = 64;
	constexpr size_t V = 50257; 

	VecBTC inputs(B);

	for(int i =0 ; i < B ; ++i){
		inputs[i] = Matf::Random(T,V);
	}

	StdVec<float> inputs_vec(B*T*V);
	for(int i = 0 ; i < B;++i){
		for(int j = 0 ; j < T ; ++j){
			for(int k = 0; k < V ; ++k){
				inputs_vec[i * T * V + j*V + k] = inputs[i](j,k); 
			}
		}
	}

	VecBTC outputs_res ;
	{
		auto start = std::chrono::high_resolution_clock::now();
		outputs_res = Softmax(inputs);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

    StdVec<float> outputs_vec(B*T*V);

	{
		auto start = std::chrono::high_resolution_clock::now();
        gpt2cuda::BatchSoftmaxForward(outputs_vec.data(),inputs_vec.data(),B,T,V,V);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

    //test outputs
    for(int i = 0 ; i < B ; ++i){
        Eigen::Map<MatfRow> map_outputs(outputs_vec.data() + i * T*V,T,V);
		EXPECT_TRUE(map_outputs.isApprox(outputs_res[i], 0.001)) 
			<< "---gpu---\n"<<map_outputs.block<3,3>(0,0) << std::endl 
			<< " ---cpu--- \n" << outputs_res[i].block<3,3>(0,0) <<std::endl ;
			// << "---gpu---\n"<<map_outputs << std::endl 
			// << " ---cpu--- \n" << outputs_res[i] <<std::endl ;

    }
}




TEST(CudaSoftmax,padding){

    //too large cpu will slow
	constexpr size_t B = 4;
	constexpr size_t T = 64;
	constexpr size_t V = 757; 
	constexpr size_t Vp = 768; 


	VecBTC inputs(B);

	for(int i =0 ; i < B ; ++i){
		inputs[i] = Matf::Random(T,Vp);
	}

	StdVec<float> inputs_vec(B*T*Vp);
	for(int i = 0 ; i < B;++i){
		for(int j = 0 ; j < T ; ++j){
			for(int k = 0; k < Vp ; ++k){
				inputs_vec[i * T * Vp + j*Vp + k] = inputs[i](j,k); 
			}
		}
	}

	VecBTC outputs_res(B);
	{
		auto start = std::chrono::high_resolution_clock::now();
        for (int b = 0; b < B; ++b) {
            outputs_res[b] = Softmax(inputs[b].block(0, 0, T, V));
        }
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

    StdVec<float> outputs_vec(B*T*Vp);

	{
		auto start = std::chrono::high_resolution_clock::now();
        gpt2cuda::BatchSoftmaxForward(outputs_vec.data(),inputs_vec.data(),B,T,V,Vp);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

    //test outputs
    for(int i = 0 ; i < B ; ++i){
        Eigen::Map<MatfRow> map_outputs(outputs_vec.data() + i * T*Vp,T,Vp);
		EXPECT_TRUE(map_outputs.block(0,0,T,V).isApprox(outputs_res[i].block(0,0,T,V), 0.001)) 
			<< "---gpu---\n"<<map_outputs.block<3,3>(0,0) << std::endl 
			<< " ---cpu--- \n" << outputs_res[i].block<3,3>(0,0) <<std::endl ;
			// << "---gpu---\n"<<map_outputs << std::endl 
			// << " ---cpu--- \n" << outputs_res[i] <<std::endl ;

    }
    
}