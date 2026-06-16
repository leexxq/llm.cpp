#include "cuda/matmul.cuh"
#include "gelu.h"
#include "global.h"
#include "matmul.h"
#include <Eigen/src/Core/util/Constants.h>
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <chrono>

using MatRow = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
TEST(CudaMatMul, forward1){
	StdVec<float> weight = {
		 1.6394, -2.9037, -1.0096, -0.4072, 0.5853 ,
		 0.7580, 0.9567, -0.9137, -0.6394, -0.4959 ,
		-1.3069, 0.6839, 1.1314, -0.0770, 0.8897 ,
		-0.0154, 0.7099, -0.7526, 0.8421, -1.2247 ,
	};

	StdVec<float> bias{-1.3034, -0.2438, 0.8006, -1.0919, 1.3184};



    StdVec<float> inputs{
        -0.6913, 1.6103, 0.1138, 0.3595 , 
        0.2048, -0.8102, -1.1874, -0.0937 , 
        1.1244, 0.5758, -0.0924, -0.8170 ,
		//
        -1.9811, 1.1705, -0.0687, -1.4556 ,  
        0.0356, -0.5523, 1.0658, -0.6316 ,
        1.2820, 0.3895, 0.0981, 0.0935 ,
    };



	// no bias 
	//    -0.066971369087696075    3.8809387683868408  -0.91520094871520996  -0.45415607094764709   -1.5421973466873169
	//    1.2748736143112183   -2.2483763694763184  -0.73939210176467896   0.44717231392860413  -0.42002773284912109
	//    2.4131371974945068   -3.3572332859039307   -1.1509698629379272   -1.5069030523300171    1.2909437417984009

	VecBTC resexp{
		Matf{
				{ -1.3704, 3.6371, -0.1147, -1.5461, -0.2238 },
				{ -0.0284, -2.4923, 0.0611, -0.6447, 0.8983 },
				{ 1.1098, -3.6011, -0.3504, -2.5988, 2.6093 },
		},
		Matf{
				{ -3.5517, 5.5484, 2.7489, -2.2541, 1.2999 },
				{ -3.0468, -0.5951, 2.9504, -1.3673, 3.3349 },
				{ 0.9637, -3.4602, -0.8088, -1.7918, 1.8485 },
		}
	};

    StdVec<float> outputs(2*3*5,0);
    gpt2cuda::BatchMatmulForward(outputs.data(),inputs.data(),weight.data(),bias.data(),2,3,4,5);

    Eigen::Map<MatRow> map_output_1(outputs.data(),3,5);
    Eigen::Map<MatRow> map_output_2(outputs.data() + 3 * 5,3,5);


		EXPECT_TRUE(map_output_1.isApprox(resexp[0], 0.001)) << 
    map_output_1 << std::endl;

		EXPECT_TRUE(map_output_2.isApprox(resexp[1], 0.001)) << 
    map_output_2 << std::endl;

}


TEST(CudaMatMul, backward1){
	StdVec<float> weight = {
		0.5496, -1.0675, -0.3498, -1.7053, -0.0094,
		0.8118, 0.6951, 1.7514, -0.0827, -0.7742,
		0.6460, -0.5852, 1.0690, 2.4317, -1.1704,
		0.0139, -0.5536, -0.0932, 0.6729, -0.1479
	};

	StdVec<float> bias{ -0.5602, -0.6464, -0.2690, 0.5687, -0.1302 };



    StdVec<float> inputs{
		1.9170, 0.7433, -1.6831, -0.5770,
		0.5002, -0.1631, -0.6377, 0.1790,
		0.8014, 0.9574, 0.5211, 0.6669,

		0.0507, -0.7415, 0.7975, -0.1952,
		-0.0915, -0.2075, -0.6837, -1.4144,
		-1.0925, -0.9400, 2.0981, 2.5490
    };


	VecBTC resexp{2,Matf(3,4)};
	resexp[0] << -2.5823, 2.4014, 2.3912, -0.1079,
			-2.5823, 2.4014, 2.3912, -0.1079,
			-2.5823, 2.4014, 2.3912, -0.1079;

	resexp[1] << -2.5823, 2.4014, 2.3912, -0.1079,
			-2.5823, 2.4014, 2.3912, -0.1079,
			-2.5823, 2.4014, 2.3912, -0.1079;

	StdVec<float> d_outputs(2*3*5,1);
	StdVec<float> d_inputs(2*3*4,0);
	StdVec<float> d_weight(4*5,0);
	StdVec<float> d_bias(5,0);

    gpt2cuda::BatchMatmulBackward(d_inputs.data(),d_weight.data(),d_bias.data(),d_outputs.data(),inputs.data(),weight.data(),2,3,4,5);

    Eigen::Map<MatRow> map_d_inputs_0(d_inputs.data(),3,4);
    Eigen::Map<MatRow> map_d_inputs_1(d_inputs.data() + 3 * 4,3,4);

	EXPECT_TRUE(map_d_inputs_0.isApprox(resexp[0], 0.001)) << 
		map_d_inputs_0 << std::endl;

	EXPECT_TRUE(map_d_inputs_1.isApprox(resexp[1], 0.001)) << 
		map_d_inputs_1 << std::endl;

	Matf d_weightexp(4, 5);
	d_weightexp << 2.0853, 2.0853, 2.0853, 2.0853, 2.0853,
			-0.3514, -0.3514, -0.3514, -0.3514, -0.3514,
			0.4122, 0.4122, 0.4122, 0.4122, 0.4122,
			1.2083, 1.2083, 1.2083, 1.2083, 1.2083;

    Eigen::Map<MatRow> map_d_weight(d_weight.data(),4,5);
		EXPECT_TRUE(map_d_weight.isApprox(d_weightexp, 0.001)) << 
    map_d_weight << std::endl;

	Vecf d_biasexp(5);
	d_biasexp << 6., 6., 6., 6., 6.;

    Eigen::Map<Eigen::Vector<float,5>> map_d_bias(d_bias.data(),5);
		EXPECT_TRUE(map_d_bias.isApprox(d_biasexp, 0.001)) << 
    map_d_bias << std::endl;

}

TEST(CudaMatMul,backward2){
	constexpr int B = 64;
	constexpr int T = 1024;
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
		gpt2cuda::BatchMatmulBackward(d_inputs_vec.data(), d_weight_vec.data(), d_bias_vec.data(), d_outputs_vec.data(), inputs_vec.data(),weight_vec.data(), B,T,C,Oc);
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
		Eigen::Map<MatRow> map_d_input(d_inputs_vec .data() + i *T*C,T,C);
		EXPECT_TRUE(map_d_input.isApprox(d_inputs_res[i], 0.001)) 
			<< "---gpu---\n"<<map_d_input.block<10,10>(T-10,C-10) << std::endl 
			<< " ---cpu--- \n" << d_inputs_res[i].block<10,10>(T-10,C-10) <<std::endl ;
		// std::ofstream log{"log.txt",std::ios::app};

		// log << map_d_input << std::endl;
	}

	//test d_weight_vec;
	Eigen::Map<MatRow> map_d_weight(d_weight_vec .data() ,C,Oc);
	EXPECT_TRUE(map_d_weight.isApprox(matmul.d_weight, 0.001));
	// EXPECT_TRUE(map_d_weight.isApprox(matmul.d_weight, 0.001)) << map_d_weight << std::endl << matmul.d_weight << std::endl;

	//test d_bias_vec;
	Eigen::Map<Vecf> map_d_bias(d_bias_vec .data() ,Oc);
	// EXPECT_TRUE(map_d_bias.isApprox(matmul.d_bias, 0.001)) ;
	EXPECT_TRUE(map_d_bias.isApprox(matmul.d_bias, 0.001)) << map_d_bias <<std::endl << matmul.d_bias << std::endl;

}

TEST(CudaMatMul,fused_gelu_forward1){
	constexpr int B = 64;
	constexpr int T = 1024;
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
		Eigen::Map<MatRow> map_output(outputs_vec .data() + i *T*Oc,T,Oc);
		EXPECT_TRUE(map_output.isApprox(outputs_res[i], 0.001)) 
			<< "---gpu---\n"<<map_output.block<10,10>(T-10,C-10) << std::endl 
			<< " ---cpu--- \n" << outputs_res[i].block<10,10>(T-10,C-10) <<std::endl ;

		// std::ofstream log{"log.txt",std::ios::app};
		// log << map_d_input << std::endl;
	}

}