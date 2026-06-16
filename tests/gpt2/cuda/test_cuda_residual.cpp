#include "cuda/residual.cuh"
#include "residual.h"
#include "global.h"
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <chrono>


using MatRow = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

TEST(CudaResidual, forward1){

	constexpr int B = 64;
	constexpr int T = 1024;
	constexpr int C = 768; 
    auto residual = Residual(B,T,C);
	VecBTC inputs1(B);
	for(int i =0 ; i < B ; ++i){
		inputs1[i] = Matf::Random(T,C);
	}
	VecBTC inputs2(B);
	for(int i =0 ; i < B ; ++i){
		inputs2[i] = Matf::Random(T,C);
	}

	StdVec<float> inputs1_vec(B*T*C);
	StdVec<float> inputs2_vec(B*T*C);
	for(int i = 0 ; i < B;++i){
		for(int j = 0 ; j < T ; ++j){
			for(int k = 0; k < C ; ++k){
				inputs1_vec[i * T * C + j*C + k] = inputs1[i](j,k); 
				inputs2_vec[i * T * C + j*C + k] = inputs2[i](j,k); 
			}
		}
	}

	VecBTC outputs_res ;
	{
		auto start = std::chrono::high_resolution_clock::now();
		outputs_res = residual.Forward(inputs1,inputs2);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

    StdVec<float> outputs_vec(B*T*C);

	{
		auto start = std::chrono::high_resolution_clock::now();
        gpt2cuda::BatchResidualForward(outputs_vec.data(),inputs1_vec.data(),inputs2_vec.data(),B,T,C);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}
    //test outputs
    for(int i = 0 ; i < B ; ++i){
        Eigen::Map<MatRow> map_outputs(outputs_vec.data() + i * T*C,T,C);
		EXPECT_TRUE(map_outputs.isApprox(outputs_res[i], 0.001)) 
			<< "---gpu---\n"<<map_outputs.block<3,3>(0,0) << std::endl 
			<< " ---cpu--- \n" << outputs_res[i].block<3,3>(0,0) <<std::endl ;
			// << "---gpu---\n"<<map_outputs << std::endl 
			// << " ---cpu--- \n" << outputs_res[i] <<std::endl ;

    }

}