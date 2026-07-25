#include "cuda/layernorm.cuh"
#include "layernorm.h"
#include "global.h"
#include <gtest/gtest.h>
#include <chrono>


using MatfRow = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

TEST(CudaLayerNorm, forward1){

	constexpr int B = 4;
	constexpr int T = 64;
	constexpr int C = 768; 
    auto layernorm = LayerNorm(B,T,C);

	layernorm.beta.setRandom(C);
	layernorm.gamma.setRandom(C);
	
	VecBTC inputs(B);
	for(int i =0 ; i < B ; ++i){
		inputs[i] = Matf::Random(T,C);
	}

	StdVec<float> inputs_vec(B*T*C);
	for(int i = 0 ; i < B;++i){
		for(int j = 0 ; j < T ; ++j){
			for(int k = 0; k < C ; ++k){
				inputs_vec[i * T * C + j*C + k] = inputs[i](j,k); 
			}
		}
	}

	VecBTC outputs_res ;
	{
		auto start = std::chrono::high_resolution_clock::now();
		outputs_res = layernorm.Forward(inputs);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

    StdVec<float> outputs_vec(B*T*C);
    StdVec<float> rstd_vec(B*T);
    StdVec<float> mean_vec(B*T);
	{
		auto start = std::chrono::high_resolution_clock::now();
        gpt2cuda::BatchLayerNormForward(outputs_vec.data(),mean_vec.data(),rstd_vec.data(),inputs_vec.data(),layernorm.gamma.data(),layernorm.beta.data(),B,T,C);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}
    //test outputs
    for(int i = 0 ; i < B ; ++i){
        Eigen::Map<MatfRow> map_outputs(outputs_vec.data() + i * T*C,T,C);
		EXPECT_TRUE(map_outputs.isApprox(outputs_res[i], 0.001)) 
			<< "---gpu---\n"<<map_outputs.block<3,3>(0,0) << std::endl 
			<< " ---cpu--- \n" << outputs_res[i].block<3,3>(0,0) <<std::endl ;
			// << "---gpu---\n"<<map_outputs << std::endl 
			// << " ---cpu--- \n" << outputs_res[i] <<std::endl ;

    }
    //test means
    for(int i = 0;i < B; ++i){
        Eigen::Map<Vecf> map_means(mean_vec.data() + i * T,T);
        EXPECT_TRUE(map_means.isApprox(layernorm.mean[i], 0.001)) 
            << "---gpu---\n"<<map_means << std::endl 
            << " ---cpu--- \n" << layernorm.mean[i] <<std::endl ;
    }

    //test rstds

    for(int i =0;i < B; ++i){
        Eigen::Map<Vecf> map_rstds(rstd_vec.data() + i*T,T);
        EXPECT_TRUE(map_rstds.isApprox(layernorm.rstd[i], 0.001)) 
            << "---gpu---\n"<<map_rstds.head(3) << std::endl 
            << " ---cpu--- \n" << layernorm.rstd[i].head(3) <<std::endl ;
    }

}


TEST(CudaLayerNorm, backward1){
	//require B,T,C

	constexpr int B = 4;
	constexpr int T = 64;
	constexpr int C = 768; 
    auto layernorm = LayerNorm(B,T,C);

	//require d_outputs inputs gamma beta mean rstd 

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

	layernorm.beta = Vecf::Random(C);
	layernorm.gamma = Vecf::Random(C);

	VecBTC d_inputs_res ;
	{
		auto _  = layernorm.Forward(inputs); //for compute means and rstds which is intermediate value used for backward 
		auto start = std::chrono::high_resolution_clock::now();
		d_inputs_res = layernorm.Backward(d_outputs, inputs);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

	StdVec<float> mean_vec(B*T);
	StdVec<float> rstd_vec(B*T);
	for(int i =0 ; i < B ; ++i){
		for(int j =0 ; j < T ; ++j){
			mean_vec[i * T + j] = layernorm.mean[i](j);
			rstd_vec[i * T + j] = layernorm.rstd[i](j);
		}
	}


    StdVec<float> d_inputs_vec(B*T*C,0);
	StdVec<float> d_gamma_vec(C,0);
	StdVec<float> d_beta_vec(C,0);

	{
		auto start = std::chrono::high_resolution_clock::now();
		
        gpt2cuda::BatchLayerNormBackward(d_inputs_vec.data(), d_gamma_vec.data(),d_beta_vec.data(),d_outputs_vec.data(), inputs_vec.data(),layernorm.gamma.data(),mean_vec.data(),rstd_vec.data(),B,T,C);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}
	

	//require 

	//test d_inputs
    for(int i = 0 ; i < B ; ++i){
        Eigen::Map<MatfRow> map_d_inputs(d_inputs_vec.data() + i * T*C,T,C);
		EXPECT_TRUE(map_d_inputs.isApprox(d_inputs_res[i], 0.001)) 
			<< "---gpu---\n"<<map_d_inputs.block<5,5>(0,0) << std::endl 
			<< " ---cpu--- \n" << d_inputs_res[i].block<5,5>(0,0) <<std::endl ;

    }

	//test d_gamma
	Eigen::Map<Vecf> map_d_gamma(d_gamma_vec.data(),C);
	EXPECT_TRUE(map_d_gamma.isApprox(layernorm.d_gamma, 0.001)) 
		<< "---gpu---\n"<<map_d_gamma.tail(9) << std::endl 
		<< " ---cpu--- \n" << layernorm.d_gamma.tail(9) <<std::endl ;

	//test d_beta
	Eigen::Map<Vecf> map_d_beta(d_beta_vec.data(),C);
		EXPECT_TRUE(map_d_beta.isApprox(layernorm.d_beta, 0.001)) 
		<< "---gpu---\n"<<map_d_beta.tail(9) << std::endl 
		<< " ---cpu--- \n" << layernorm.d_beta.tail(9) <<std::endl ;
}