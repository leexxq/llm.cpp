#include "cuda/encoder.cuh"
#include "encoder.h"
#include "global.h"
#include <gtest/gtest.h>
#include <cassert>
#include <chrono>
#include <random>

using MatfRow = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using MatiRow = Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;


TEST(CudaEncoder, forward1){
	constexpr int B = 64;
	constexpr int T = 1024;
	constexpr int Vp = 50304;
	constexpr int V = 50257;
	constexpr int MaxT = 1024;
	constexpr int C = 768; 
    auto encoder = Encoder(Vp,MaxT,C);
    assert(T <= MaxT);



    encoder.wte.setRandom(Vp,C);
    encoder.wpe.setRandom(MaxT,C);


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, V-1); // Bounded range [0, V]

    // Create a 4x4 matrix using a lambda expression
    Mati inputs = Mati::NullaryExpr(B, T, [&]() { 
        return dist(gen); 
    });


    StdVec<int> inputs_vec(B*T);

    Eigen::Map<MatiRow> (inputs_vec.data(),B,T) = inputs;
    // std::cout << "gpu:" << std::endl;

    // for(auto & v : inputs_vec){
    //     std::cout << v << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "cpu:" << std::endl;
    // std::cout << inputs << std::endl;

	VecBTC outputs_res;
	{
		auto start = std::chrono::high_resolution_clock::now();
		outputs_res = encoder.Forward(inputs);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

    StdVec<float> outputs_vec(B*T*C);
    StdVec<float> wte_vec(Vp*C);
    StdVec<float> wpe_vec(MaxT*C);
    Eigen::Map<MatfRow>(wte_vec.data(),Vp,C) = encoder.wte;
    Eigen::Map<MatfRow>(wpe_vec.data(),MaxT,C) = encoder.wpe;


	{
		auto start = std::chrono::high_resolution_clock::now();
        gpt2cuda::BatchEncoderForward(outputs_vec.data(),inputs_vec.data(),wte_vec.data(),wpe_vec.data(),B,T,C,Vp,MaxT);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}


    //test outputs
    for(int i = 0 ; i < B ; ++i){
        Eigen::Map<MatfRow> map_outputs(outputs_vec.data() + i * T*C,T,C);
		EXPECT_TRUE(map_outputs.isApprox(outputs_res[i], 0.001f)) 
			<< "---gpu---\n"<<map_outputs.block<3,3>(0,0) << std::endl 
			<< " ---cpu--- \n" << outputs_res[i].block<3,3>(0,0) <<std::endl ;

    }

}



TEST(CudaEncoder, backward1){

	constexpr int B = 64;
	constexpr int T = 1024;
	constexpr int Vp = 50304;
	constexpr int V = 50257;
	constexpr int MaxT = 1024;
	constexpr int C = 768; 
    auto encoder = Encoder(Vp,MaxT,C);


    encoder.wte.setRandom(Vp,C);
    encoder.wpe.setRandom(MaxT,C);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, V-1); // Bounded range [0, V]

    // Create a 4x4 matrix using a lambda expression
    Mati inputs = Mati::NullaryExpr(B, T, [&]() { 
        return dist(gen); 
    });

    StdVec<int> inputs_vec(B*T);

    Eigen::Map<MatiRow>(inputs_vec.data(),B,T) = inputs;




	VecBTC d_outputs(B);
	for(int i =0 ; i < B ; ++i){
		d_outputs[i] = Matf::Random(T,C);
	}

	StdVec<float> d_outputs_vec(B*T*C);
    for(int b =0 ; b < B ; ++b){
        Eigen::Map<MatfRow>(d_outputs_vec.data() + b *T*C,T,C) = d_outputs[b];
    }

	{
		auto start = std::chrono::high_resolution_clock::now();
        encoder.Backward(d_outputs, inputs);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

	StdVec<float> d_wte_vec(Vp*C,0);
	StdVec<float> d_wpe_vec(MaxT * C,0);
    
    {
		auto start = std::chrono::high_resolution_clock::now();
        gpt2cuda::BatchEncoderBackward(d_wte_vec.data(),d_wpe_vec.data(),d_outputs_vec.data(),inputs_vec.data(),B,T,C,Vp,MaxT);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}


    //test dwte
    Eigen::Map<MatfRow> map_d_wte(d_wte_vec.data(),Vp,C);
    EXPECT_TRUE(map_d_wte .isApprox(encoder.d_wte))
		<< "---gpu---\n"<<map_d_wte.block(0,0,3,3) << std::endl 
		<< " ---cpu--- \n" << encoder.d_wte.block(0,0,3,3) <<std::endl ;

    //test_dwpe
    Eigen::Map<MatfRow> map_d_wpe(d_wpe_vec.data(),MaxT,C);
    EXPECT_TRUE(map_d_wpe .isApprox(encoder.d_wpe))
		<< "---gpu---\n"<<map_d_wpe.block(0,0,3,3) << std::endl 
		<< " ---cpu--- \n" << encoder.d_wpe.block(0,0,3,3) <<std::endl ;
}
