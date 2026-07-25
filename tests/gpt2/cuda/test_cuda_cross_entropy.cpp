#include "cuda/cross_entropy.cuh"
#include "cross_entropy.h"
#include "global.h"
#include "softmax.h"
#include <gtest/gtest.h>
#include <chrono>
#include <numeric>
#include <random>


using MatfRow = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using MatiRow = Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;


TEST(CudaCrossEntropy, forward1){
	constexpr size_t B = 4;
	constexpr size_t T = 64;
	constexpr size_t Vp = 50304;
	constexpr size_t V = 50257;

    auto loss = CrossEntropy(B,T);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, V-1); // Bounded range [0, V]

    // Create a 4x4 matrix using a lambda expression
    Mati targets = Mati::NullaryExpr(B, T, [&]() { 
        return dist(gen); 
    });

    StdVec<int> targets_vec(B*T);
    Eigen::Map<MatiRow>(targets_vec.data(),B,T) = targets;

    VecBTVp probs(B);
    StdVec<float> probs_vec(B*T*Vp);
    auto iter = probs_vec.begin();
    for(auto& prob : probs){
        prob = Matf::Random(T,Vp);
        Softmax(prob,prob);
        // prob.rowwise().norm();
        Eigen::Map<MatfRow> (iter.base(),T,Vp) = prob;
        iter += T*Vp;
    }
    
    
    float mean_loss_exp;
	{
		auto start = std::chrono::high_resolution_clock::now();
		mean_loss_exp = loss.Forward(probs, targets);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

    
    StdVec<float> losses(B*T);

    float mean_loss;
	{
		auto start = std::chrono::high_resolution_clock::now();
        gpt2cuda::BatchCrossEntropyForward(losses.data(),probs_vec.data(),targets_vec.data(),B,T,Vp);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        mean_loss = std::accumulate(losses.begin(),losses.end(),0.f);
        mean_loss /= B*T;
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

    EXPECT_NEAR(mean_loss, mean_loss_exp, 0.001f);

    //test losses
    for(int b = 0 ; b < B ; ++b){
        Eigen::Map<Vecf> map_losses(losses.data() + b*T,T);
        EXPECT_TRUE(map_losses.isApprox(loss.losses[b],0.001f))
            << "---gpu---\n"<<map_losses.tail(9) << std::endl 
			<< " ---cpu--- \n" << loss.losses[b].tail(9)<<std::endl ;
    }
}

TEST(CudaCrossEntropy, fused_softmax_backward1){
	constexpr size_t B = 4;
	constexpr size_t T = 64;
	constexpr size_t Vp = 50304;
	constexpr size_t V = 50257;


    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, V-1); // Bounded range [0, V]

    // Create a 4x4 matrix using a lambda expression
    Mati targets = Mati::NullaryExpr(B, T, [&]() { 
        return dist(gen); 
    });

    StdVec<int> targets_vec(B*T);
    Eigen::Map<MatiRow>(targets_vec.data(),B,T) = targets;

    VecBTVp probs(B);
    StdVec<float> probs_vec(B*T*Vp);
    {
        auto iter = probs_vec.begin();
        for(auto& prob : probs){
            prob = Matf::Random(T,Vp);
            Softmax(prob,prob);
            Eigen::Map<MatfRow> (iter.base(),T,Vp) = prob;
            iter+=T*Vp;
        }
    }

	VecBTC d_inputs(B);
	StdVec<float> d_inputs_vec(B*T*Vp);
    {
        auto iter = d_inputs_vec.begin();
        for(auto&d_input:d_inputs){
            d_input = Matf::Zero(T,Vp);
            Eigen::Map<MatfRow>(iter.base(),T,Vp) = d_input;
            iter +=T*Vp;
        }
    }

	{
		auto start = std::chrono::high_resolution_clock::now();
        CrossEntropySoftmaxBackward(d_inputs, probs, targets);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

    {
		auto start = std::chrono::high_resolution_clock::now();
        gpt2cuda::BatchCrossEntropySoftmaxBackward(d_inputs_vec.data(), probs_vec.data(), targets_vec.data(),B,T,V,Vp,1.f/(B*T));
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

    //test outputs
    for(int i = 0 ; i < B ; ++i){
        Eigen::Map<MatfRow> map_d_inputs(d_inputs_vec.data() + i * T*Vp,T,Vp);
		EXPECT_TRUE(map_d_inputs.block(0,0,T,V).isApprox(d_inputs[i].block(0,0,T,V), 0.001f)) 
			<< "---gpu---\n"<<map_d_inputs.block<3,3>(0,0) << std::endl 
			<< " ---cpu--- \n" << d_inputs[i].block<3,3>(0,0) <<std::endl ;

    }

    
}