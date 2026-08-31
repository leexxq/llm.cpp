#include "cuda/matmul_softmax.cuh"
#include "cuda/matmul.cuh"
#include "gelu.h"
#include "global.h"
#include "matmul.h"
#include "softmax.h"

#include <Eigen/src/Core/util/Constants.h>
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <chrono>
#include <fstream>

using MatfRow = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

TEST(CudaMatMul, forward1){
	constexpr int B = 4;
	constexpr int T = 512;
	constexpr int C = 768; 
	constexpr int Oc = 4 * C;

	auto matmul = MatMul(C,Oc);

	matmul.weight = Matf::Random(C,Oc);
	matmul.bias = Vecf::Random(Oc);

	VecBTC inputs(B);
	StdVec<float> inputs_vec(B * T * C);
	for(int i =0 ; i < B ; ++i){
		inputs[i] = Matf::Random(T,C);
		Eigen::Map<MatfRow> (inputs_vec.data() + i * T*C,T,C) = inputs[i];
	}

	VecBTC outputs_res;
	{
		auto start = std::chrono::high_resolution_clock::now();
		outputs_res = matmul.Forward(inputs);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

    StdVec<float> outputs_vec(B*T*Oc,0);
	{
		auto start = std::chrono::high_resolution_clock::now();
		gpt2cuda::BatchMatmulNTForward(outputs_vec.data(),inputs_vec.data(),matmul.weight.data(),matmul.bias.data(),B,T,C,Oc);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

	for(int b = 0 ; b < B ; ++ b){
		Eigen::Map<MatfRow> map_output_vec(outputs_vec.data() + b * T * Oc,T,Oc);
			EXPECT_TRUE(map_output_vec.isApprox(outputs_res[b], 0.001))
			 << "gpu" << std::endl << map_output_vec.block<3,3>(0,0) << std::endl 
			<< "cpu" << std::endl <<  outputs_res[b].block<3,3> (0,0) << std::endl;
	}


}


TEST(CudaMatMul,backward1){
	constexpr int B = 4;
	constexpr int T = 512;
	constexpr int C = 768; 
	constexpr int Oc = 4 * C;
	VecBTC inputs(B);
	VecBTC d_outputs(B);
	for(int i =0 ; i < B ; ++i){
		inputs[i] = Matf::Random(T,C);
		d_outputs[i] = Matf::Random(T,Oc);
	}

	auto matmul = MatMul(C,Oc);

	matmul.weight = Matf::Random(C,Oc);

	StdVec<float> weight_vec(C*Oc);
	for(int i = 0; i < C ; ++i){
		for(int j = 0 ; j < Oc ; ++j){
			weight_vec.data()[i*Oc + j] = matmul.weight(i,j);
		}
	}

	matmul.bias = Vecf::Random(Oc);

	VecBTC d_inputs_res ;
	{
		auto start = std::chrono::high_resolution_clock::now();
		d_inputs_res = matmul.Backward(d_outputs, inputs);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

	StdVec<float> inputs_vec(B*T*C);
	for(int i = 0 ; i < B;++i){
		for(int j = 0 ; j < T ; ++j){
			for(int k = 0; k < C ; ++k){
				inputs_vec[i * T * C + j*C + k] = inputs[i](j,k); 
			}
		}
	}
	StdVec<float> d_outputs_vec(B*T*Oc);
	for(int i = 0 ; i < B;++i){
		for(int j = 0 ; j < T ; ++j){
			for(int k =0 ; k < Oc ; ++k){
				d_outputs_vec[i * T*Oc + j * Oc +k] = d_outputs[i](j,k);
			}
		}
	}

	StdVec<float> d_bias_vec(Oc,0);

	StdVec<float> d_weight_vec(C*Oc,0);

	StdVec<float> d_inputs_vec(B*T*C,0);

	{
		auto start = std::chrono::high_resolution_clock::now();
		gpt2cuda::BatchMatmulNNBackward(d_inputs_vec.data(), d_weight_vec.data(), d_bias_vec.data(), d_outputs_vec.data(), inputs_vec.data(),weight_vec.data(), B,T,C,Oc);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

	// Eigen::Map<MatRow> map_inputs_vec(inputs_vec.data(),T,C);
	// std::ofstream log{"log.txt",std::ios::trunc};
	// log << map_inputs_vec << std::endl;

	//test d_inputs_vec;
	for(int i =0 ; i < B ; ++i){
		// for(int j=0; j < T; ++j){
		// 	for(int k=0; k < C; ++k){
		// 		EXPECT_NEAR(d_inputs_vec[i*T*C +j*C + k] ,d_inputs_res[i](j,k),0.001f);
		// 	}
		// }
		Eigen::Map<MatfRow> map_d_input(d_inputs_vec .data() + i *T*C,T,C);
		EXPECT_TRUE(map_d_input.isApprox(d_inputs_res[i], 0.001)) 
			<< "---gpu---\n"<<map_d_input.block<10,10>(T-10,C-10) << std::endl 
			<< " ---cpu--- \n" << d_inputs_res[i].block<10,10>(T-10,C-10) <<std::endl ;
		// std::ofstream log{"log.txt",std::ios::app};

		// log << map_d_input << std::endl;
	}

	//test d_weight_vec;
	Eigen::Map<MatfRow> map_d_weight(d_weight_vec .data() ,C,Oc);
	EXPECT_TRUE(map_d_weight.isApprox(matmul.d_weight, 0.001));
	// EXPECT_TRUE(map_d_weight.isApprox(matmul.d_weight, 0.001)) << map_d_weight << std::endl << matmul.d_weight << std::endl;

	//test d_bias_vec;
	Eigen::Map<Vecf> map_d_bias(d_bias_vec .data() ,Oc);
	// EXPECT_TRUE(map_d_bias.isApprox(matmul.d_bias, 0.001)) ;
	EXPECT_TRUE(map_d_bias.isApprox(matmul.d_bias, 0.001f))
	<< "gpu:" << map_d_bias <<std::endl << "cpu: " << matmul.d_bias << std::endl;

}


TEST(CudaMatMul,backward2){
	constexpr int B = 4;
	constexpr int T = 512;
	constexpr int C = 768; 
	constexpr int Oc = 4 * C;
	VecBTC inputs(B);
	VecBTC d_outputs(B);
	StdVec<float> inputs_vec(B*T*C);
	StdVec<float> d_outputs_vec(B*T*Oc);
	for(int i =0 ; i < B ; ++i){
		inputs[i] = Matf::Random(T,C);
		d_outputs[i] = Matf::Random(T,Oc);
		Eigen::Map<MatfRow>(inputs_vec.data() + i * T * C , T , C) = inputs[i];
		Eigen::Map<MatfRow>(d_outputs_vec.data() + i * T * Oc , T , Oc) = d_outputs[i];
	}

	auto matmul = MatMul(C,Oc);

	matmul.weight = Matf::Random(C,Oc);


	matmul.bias = Vecf::Random(Oc);

	VecBTC d_inputs_res ;
	{
		auto start = std::chrono::high_resolution_clock::now();
		d_inputs_res = matmul.Backward(d_outputs, inputs);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}


	StdVec<float> d_bias_vec(Oc,0);

	StdVec<float> d_weight_vec(C*Oc,0);

	StdVec<float> d_inputs_vec(B*T*C,0);

	{
		auto start = std::chrono::high_resolution_clock::now();
		gpt2cuda::BatchMatmulNTBackward(d_inputs_vec.data(), d_weight_vec.data(), d_bias_vec.data(), d_outputs_vec.data(), inputs_vec.data(),matmul.weight.data(), B,T,C,Oc);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

	// Eigen::Map<MatRow> map_inputs_vec(inputs_vec.data(),T,C);
	// std::ofstream log{"log.txt",std::ios::trunc};
	// log << map_inputs_vec << std::endl;

	//test d_inputs_vec;
	for(int i =0 ; i < B ; ++i){
		// for(int j=0; j < T; ++j){
		// 	for(int k=0; k < C; ++k){
		// 		EXPECT_NEAR(d_inputs_vec[i*T*C +j*C + k] ,d_inputs_res[i](j,k),0.001f);
		// 	}
		// }
		Eigen::Map<MatfRow> map_d_input(d_inputs_vec .data() + i *T*C,T,C);
		EXPECT_TRUE(map_d_input.isApprox(d_inputs_res[i], 0.001)) 
			<< "---gpu---\n"<<map_d_input.block<10,10>(T-10,C-10) << std::endl 
			<< " ---cpu--- \n" << d_inputs_res[i].block<10,10>(T-10,C-10) <<std::endl ;
		// std::ofstream log{"log.txt",std::ios::app};

		// log << map_d_input << std::endl;
	}

	//test d_weight_vec;
	Eigen::Map<Matf> map_d_weight(d_weight_vec .data() ,Oc,C);
	// EXPECT_TRUE(map_d_weight.isApprox(matmul.d_weight, 0.001));
	EXPECT_TRUE(map_d_weight.isApprox(matmul.d_weight, 0.001)) 
	<< "gpu:\n"<< map_d_weight.block<3,3>(0,0) << std::endl 
	<< "cpu:\n"<< matmul.d_weight.block<3,3>(0,0) << std::endl;

	//test d_bias_vec;
	Eigen::Map<Vecf> map_d_bias(d_bias_vec .data() ,Oc);
	// EXPECT_TRUE(map_d_bias.isApprox(matmul.d_bias, 0.001)) ;
	EXPECT_TRUE(map_d_bias.isApprox(matmul.d_bias, 0.001f))
	<< "gpu:" << map_d_bias <<std::endl << "cpu: " << matmul.d_bias << std::endl;

}

TEST(CudaMatMul,fused_gelu_forward1){
	constexpr int B = 4;
	constexpr int T = 64;
	constexpr int C = 768; 
	constexpr int Oc = 4 * C;
	VecBTC inputs(B);
	for(int i =0 ; i < B ; ++i){
		inputs[i] = Matf::Random(T,C);
	}

	auto matmul = MatMul(C,Oc);
	auto gelu = GELU();

	matmul.weight = Matf::Random(C,Oc);

	StdVec<float> weight_vec(C*Oc);
	for(int i = 0; i < C ; ++i){
		for(int j = 0 ; j < Oc ; ++j){
			weight_vec.data()[i*Oc + j] = matmul.weight(i,j);
		}
	}

	matmul.bias = Vecf::Random(Oc);

	VecBTC outputs_res ;
	{
		auto start = std::chrono::high_resolution_clock::now();
		outputs_res = gelu.Forward(matmul.Forward(inputs));
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

	StdVec<float> inputs_vec(B*T*C);
	for(int i = 0 ; i < B;++i){
		for(int j = 0 ; j < T ; ++j){
			for(int k = 0; k < C ; ++k){
				inputs_vec[i * T * C + j*C + k] = inputs[i](j,k); 
			}
		}
	}

	StdVec<float> outputs_vec(B*T*Oc);

	{
		auto start = std::chrono::high_resolution_clock::now();
		gpt2cuda::BatchMatmulGeluForward(outputs_vec.data(),inputs_vec.data(), weight_vec.data(), matmul.bias.data(), B,  T,  C,  Oc);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

	for(int i =0 ; i < B ; ++i){
		Eigen::Map<MatfRow> map_output(outputs_vec .data() + i *T*Oc,T,Oc);
		EXPECT_TRUE(map_output.isApprox(outputs_res[i], 0.001)) 
			<< "---gpu---\n"<<map_output.block<10,10>(T-10,C-10) << std::endl 
			<< " ---cpu--- \n" << outputs_res[i].block<10,10>(T-10,C-10) <<std::endl ;

		// std::ofstream log{"log.txt",std::ios::app};
		// log << map_d_input << std::endl;
	}

}



TEST(CudaMatMul, fused_softmax_forward1){
	constexpr int B = 4;
	constexpr int T = 512;
	constexpr int C = 768;
	constexpr int V =  50257; 
	constexpr int Vp = 50304;

	auto matmul = MatMul(C,Vp);

	matmul.weight = Matf::Random(C,Vp);
	matmul.bias = Vecf::Random(Vp);

	VecBTC inputs(B);
	StdVec<float> inputs_vec(B * T * C);
	for(int i =0 ; i < B ; ++i){
		inputs[i] = Matf::Random(T,C);
		Eigen::Map<MatfRow> (inputs_vec.data() + i * T*C,T,C) = inputs[i];
	}

	VecBTV outputs_res(B);
	VecBTVp logits;
	{
		auto start = std::chrono::high_resolution_clock::now();
		logits = matmul.Forward(inputs);
		for (int b = 0; b < B; ++b) {
			outputs_res[b] = Softmax(logits[b].block(0, 0, T, V));
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

    StdVec<float> outputs_vec(B*T*Vp);
    StdVec<float> soft_outputs_vec(B*T*Vp);
	
	{
		auto start = std::chrono::high_resolution_clock::now();
		gpt2cuda::BatchMatmulNTSoftmaxForward(soft_outputs_vec.data(), outputs_vec.data(),inputs_vec.data(),matmul.weight.data(),matmul.bias.data(),B,T,C,Vp,V);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}
	std::fstream log{"matmul_softamx_forward_log" , std::ios::out | std::ios::trunc};

	for(int b = 0 ; b < B ; ++ b){
		Eigen::Map<MatfRow> map_soft_output_vec(soft_outputs_vec.data() + b * T * Vp,T,Vp);
			EXPECT_TRUE(map_soft_output_vec.block(0,0,T,V).isApprox(outputs_res[b], 0.1f))
			 << "gpu" << std::endl << map_soft_output_vec.block<9,9>(T - 10, V - 10) << std::endl 
			<< "cpu" << std::endl <<  outputs_res[b].block<9,9> (T - 10, V - 10) << std::endl;
			//  << "gpu" << std::endl << map_output_vec.block<9,9>(T - 10, V - 10) << std::endl 
			// << "cpu" << std::endl <<  logits[b].block<9,9> (T - 10, V - 10) << std::endl;
		
		// std::cout << map_soft_output_vec.block(0,0,T,V).rowwise().sum()<< std::endl;
		// std::cout << outputs_res[b].maxCoeff() << std::endl;

		// log << "-------" << B << "-------" << std::endl;
		// log << "-------gpu-------" << std::endl;
		// log << map_output_vec << std::endl;
		// log << "-------cpu-------" << std::endl;
		// log << outputs_res[b] << std::endl;
	}


}