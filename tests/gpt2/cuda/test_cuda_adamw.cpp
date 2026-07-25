#include "global.h"
#include "cuda/adamw.cuh"
#include "adamw.h"
#include <gtest/gtest.h>
#include <chrono>
#include <random>

using MatfRow = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using MatiRow = Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

TEST(CudaAdamW,update1){
	constexpr size_t B = 4;
	constexpr size_t T = 64;
	constexpr size_t Vp = 50304;
	constexpr size_t V = 50257;


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> real_dist0_1;
    std::uniform_int_distribution<int> int_dist1_100(1,100);

    StdVec<float> data_cpu(B*T*Vp);
    StdVec<float> grad_data(B*T*Vp);
    StdVec<float> m_cpu(B*T*Vp);
    StdVec<float> v_cpu(B*T*Vp);

    assert(data_cpu.size() == grad_data.size());
    assert(data_cpu.size() == m_cpu.size());
    assert(v_cpu.size() == m_cpu.size());
    for(size_t i = 0 ; i < data_cpu.size(); ++i){
        data_cpu[i] = real_dist0_1(gen) *100;
        grad_data[i] = real_dist0_1(gen) * 100;
        m_cpu[i] = real_dist0_1(gen);
        v_cpu[i] = real_dist0_1(gen);
    }

    

    AdamWParams adamw_params;
    adamw_params.lr = real_dist0_1(gen);
    adamw_params.beta1 = real_dist0_1(gen) * (1 - 1e-3);
    adamw_params.beta2 = real_dist0_1(gen) * (1 - 1e-3);
    adamw_params.eps= (real_dist0_1(gen) + 1e-9) * 1e-7;
    adamw_params.weight_decay= real_dist0_1(gen) / 100;
    adamw_params.t= int_dist1_100(gen);


    StdVec<float> data_gpu(data_cpu);
    StdVec<float> m_gpu(m_cpu);
    StdVec<float> v_gpu(v_cpu);

    //run on cpu
    {
		auto start = std::chrono::high_resolution_clock::now();
        AdamW(data_cpu.data(), grad_data.data(),m_cpu.data(),v_cpu.data(), m_cpu.size(), adamw_params);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

    //run on gpu
    {
		auto start = std::chrono::high_resolution_clock::now();
        gpt2cuda::AdamW(data_gpu.data(), grad_data.data(),m_cpu.data(),v_cpu.data(), m_cpu.size(), adamw_params.lr,adamw_params.beta1,adamw_params.beta2,adamw_params.eps,adamw_params.weight_decay,adamw_params.t);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

    
    //test data 
    for(int i = 0 ; i < B ; ++i){
        Eigen::Map<MatfRow> map_data_gpu(data_gpu.data() + i * T*Vp,T,Vp);
        Eigen::Map<MatfRow> map_data_cpu(data_cpu.data() + i * T*Vp,T,Vp);
		EXPECT_TRUE(map_data_gpu.isApprox(map_data_cpu, 0.001f)) 
			<< "---gpu---\n"<<map_data_gpu.block<3,3>(0,0) << std::endl 
			<< " ---cpu--- \n" << map_data_cpu.block<3,3>(0,0) <<std::endl ;

    }


}