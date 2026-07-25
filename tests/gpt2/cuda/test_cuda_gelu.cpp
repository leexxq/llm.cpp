#include "gelu.h"
#include "global.h"
#include "cuda/gelu.cuh"
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <chrono>


using MatfRow = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

TEST(CudaGelu, backward1){

    auto gelu = GELU();

	constexpr int B = 4;
	constexpr int T = 64;
	constexpr int C = 768; 
	VecBTC inputs(B);
	VecBTC d_outputs(B);
	for(int i =0 ; i < B ; ++i){
		inputs[i] = Matf::Random(T,C);
		d_outputs[i] = Matf::Random(T,C);
	}

	//convert to row-major memory
	StdVec<float> inputs_vec(B*T*C);
	for(int i = 0 ; i < B;++i){
		for(int j = 0 ; j < T ; ++j){
			for(int k = 0; k < C ; ++k){
				inputs_vec[i * T * C + j*C + k] = inputs[i](j,k); 
			}
		}
	}
	StdVec<float> d_outputs_vec(B*T*C);
	for(int i = 0 ; i < B;++i){
		for(int j = 0 ; j < T ; ++j){
			for(int k =0 ; k <  C; ++k){
				d_outputs_vec[i * T*C + j * C +k] = d_outputs[i](j,k);
			}
		}
	}

	VecBTC d_inputs_res ;
	{
		auto start = std::chrono::high_resolution_clock::now();
		d_inputs_res = gelu.Backward(d_outputs, inputs);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

    StdVec<float> d_inputs_vec(B*T*C,0);
	{
		auto start = std::chrono::high_resolution_clock::now();
		
        gpt2cuda::BatchGeluBackward(d_inputs_vec.data(), d_outputs_vec.data(), inputs_vec.data(),B,T,C);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}
    for(int i = 0 ; i < B ; ++i){
        Eigen::Map<MatfRow> map_d_inputs(d_inputs_vec.data() + i * T*C,T,C);
		EXPECT_TRUE(map_d_inputs.isApprox(d_inputs_res[i], 0.001)) 
			<< "---gpu---\n"<<map_d_inputs.block<3,3>(0,0) << std::endl 
			<< " ---cpu--- \n" << d_inputs_res[i].block<3,3>(0,0) <<std::endl ;

    }
}