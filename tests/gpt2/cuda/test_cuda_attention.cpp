#include "cuda/attention.cuh"
#include "attention.h"
#include "global.h"
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <chrono>
#include <cstddef>
#include <numeric>
#include <random>
#include "softmax.h"

using MatfRow = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using MatiRow = Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;


// used to test no mask's flash attention implement
class TestAttention : public Attention{
	public:
		VecBHTT pre_att_no_scaled;
		VecBHTT att_no_scaled;
		VecBHTT output_no_scaled;
	private:

		Attention::THc ScaledDotAttention(int b, int h, const Matf &qq, const Matf &kk, const Matf &vv) {
			size_t seq_len = vv.rows();
			size_t head_channels = vv.cols();
			float scale = 1.f / std::sqrt(head_channels);
			Matf output = Matf(seq_len, head_channels);


			pre_att_no_scaled[b][h] = qq * kk.transpose();
			pre_att[b][h] = pre_att_no_scaled[b][h] * scale;

			{
				size_t cols = pre_att[b][h].cols();
				size_t rows = pre_att[b][h].rows();
				for (int r = 0; r < rows; ++r) {
					float maxval = pre_att[b][h].row(r).maxCoeff();
					att_no_scaled[b][h].row(r) = (pre_att[b][h].row(r).array() - maxval).exp();

					att[b][h].row(r) = (pre_att[b][h].row(r).array() - maxval).exp();
					float expsum = att[b][h].row(r).sum();
					float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
					att[b][h].row(r) *= expsum_inv;
				}
			}

			output_no_scaled[b][h] = att_no_scaled[b][h] * vv;
			
			output = att[b][h]  * vv;

			return output;
		}
	public:
		TestAttention() {}
		TestAttention(size_t B, size_t T, size_t C, size_t NH) : Attention(B,T,C,NH),
			pre_att_no_scaled(makeZero(B, NH, T, T)),
			att_no_scaled(makeZero(B, NH, T, T)),
			output_no_scaled(makeZero(B, NH, T, T))
		{
	
		}
		VecBTC ForwardWithNoCausal(const VecBTC &inputs) {
			size_t batchs = inputs.size();
			size_t seq_len = inputs.front().rows();
			size_t channels_3 = inputs.front().cols();
			assert(channels_3 % 3 == 0);
			size_t channels = channels_3 / 3;
			VecBTC output(batchs, Matf(seq_len, channels));
			size_t num_heads = att.front().size();

			assert(channels % num_heads == 0);
			//it's q's dimensions each attention head that is same as k,v;
			size_t head_channels = channels / num_heads;

			for (int b = 0; b < batchs; ++b) {
				//it's Q part
				const Matf &qQ = inputs[b].block(0, 0, seq_len, channels);
				const Matf &kK = inputs[b].block(0, channels, seq_len, channels);
				const Matf &vV = inputs[b].block(0, 2 * channels, seq_len, channels);
				for (int h = 0; h < num_heads; ++h) {
					const Matf &qq = qQ.block(0, h * head_channels, seq_len, head_channels);
					const Matf &kk = kK.block(0, h * head_channels, seq_len, head_channels);
					const Matf &vv = vV.block(0, h * head_channels, seq_len, head_channels);
					output[b].block(0, h * head_channels, seq_len, head_channels) = ScaledDotAttention(b, h, qq, kk, vv);
				}
			}
			return output;
		}
};

// min attention example
TEST(CudaAttention,flash_attention_f32_forward1){
	constexpr size_t B = 1;
	constexpr size_t T = 64;
    constexpr size_t NH = 1 ;
    constexpr size_t C = 32 * NH;
    constexpr size_t C3 = 3 * C;

    auto att =  TestAttention(B,T,C,NH);

    VecBTC inputs(B);
	StdVec<float> inputs_vec(B*T*C3);
	for(int i =0 ; i < B ; ++i){
		inputs[i] = Matf::Random(T,C3);
        Eigen::Map<MatfRow> (inputs_vec.data() + i * T*C3 , T,C3)  = inputs[i];
	}



	VecBTC outputs_res ;
	{
		auto start = std::chrono::high_resolution_clock::now();
		outputs_res = att.ForwardWithNoCausal(inputs);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}


    StdVec<float> outputs_vec(B*T*C);

	{
		auto start = std::chrono::high_resolution_clock::now();
        gpt2cuda::BatchAttentionForward(outputs_vec.data(), inputs_vec.data(), B, T, C3,NH);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}


	// std::cout << "outputs_vec : " << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	auto outputs_vec_map = Eigen::Map<MatfRow>(outputs_vec.data() + b * T * C,T,C);
	// 	std::cout << outputs_vec_map << std::endl;
	// 	std::cout << "---------------\n";
	// }

	// std::cout << "Q:" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << inputs[b].block(0,0,T,C).block(0,h* C/NH,T,C/NH) << std::endl;
	// 	}
	// }

	// std::cout << "K:" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << inputs[b].block(0,C,T,C).block(0,h*C/NH,T,C/NH) << std::endl;
	// 	}
	// }

	// std::cout << "V:" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << inputs[b].block(0,2*C,T,C).block(0,h*C/NH,T,C/NH) << std::endl;
	// 	}
	// }

	// std::cout << "outputs_res : " << B << ", " << T  << ", " << C  << std::endl;
	// for(auto & v : outputs_res){
	// 	std::cout << v << std::endl;
	// }


	// std::cout << "pre_att_no_scaled(S):" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << att.pre_att_no_scaled[b][h] << std::endl;
	// 	}
	// }
	// std::cout << "att_no_scaled(e^(S-m)):" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << att.att_no_scaled[b][h] << std::endl;
	// 	}
	// }
	// std::cout << "output_no_scaled(e^(S-m) * V):" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << att.output_no_scaled[b][h] << std::endl;
	// 	}
	// }


	// 
	for(int b = 0 ; b < B ; ++b){
		auto outputs_vec_map = Eigen::Map<MatfRow>(outputs_vec.data() + b * T * C,T,C);
		EXPECT_TRUE(outputs_vec_map.isApprox(outputs_res[b], 0.01f)) 
			<< "---gpu---\n"<<outputs_vec_map.block<3,3>(0,0) << std::endl 
			<< " ---cpu--- \n" << outputs_res[b].block<3,3>(0,0) <<std::endl ;
	}

}



// multi batch  and multi head
TEST(CudaAttention,flash_attention_f32_forward2){
	constexpr size_t B = 64;
	constexpr size_t T = 64;
    constexpr size_t NH = 12 ;
    constexpr size_t C = 32 * NH;
    constexpr size_t C3 = 3 * C;

    auto att =  TestAttention(B,T,C,NH);

    VecBTC inputs(B);
	StdVec<float> inputs_vec(B*T*C3);
	for(int i =0 ; i < B ; ++i){
		inputs[i] = Matf::Random(T,C3);
        Eigen::Map<MatfRow> (inputs_vec.data() + i * T*C3 , T,C3)  = inputs[i];
	}



	VecBTC outputs_res ;
	{
		auto start = std::chrono::high_resolution_clock::now();
		outputs_res = att.ForwardWithNoCausal(inputs);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}


    StdVec<float> outputs_vec(B*T*C);

	{
		auto start = std::chrono::high_resolution_clock::now();
        gpt2cuda::BatchAttentionForward(outputs_vec.data(), inputs_vec.data(), B, T, C3,NH);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}


	// std::cout << "outputs_vec : " << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	auto outputs_vec_map = Eigen::Map<MatfRow>(outputs_vec.data() + b * T * C,T,C);
	// 	std::cout << outputs_vec_map << std::endl;
	// 	std::cout << "---------------\n";
	// }

	// std::cout << "Q:" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << inputs[b].block(0,0,T,C).block(0,h* C/NH,T,C/NH) << std::endl;
	// 	}
	// }

	// std::cout << "K:" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << inputs[b].block(0,C,T,C).block(0,h*C/NH,T,C/NH) << std::endl;
	// 	}
	// }

	// std::cout << "V:" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << inputs[b].block(0,2*C,T,C).block(0,h*C/NH,T,C/NH) << std::endl;
	// 	}
	// }

	// std::cout << "outputs_res : " << B << ", " << T  << ", " << C  << std::endl;
	// for(auto & v : outputs_res){
	// 	std::cout << v << std::endl;
	// }


	// std::cout << "pre_att_no_scaled(S):" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << att.pre_att_no_scaled[b][h] << std::endl;
	// 	}
	// }
	// std::cout << "att_no_scaled(e^(S-m)):" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << att.att_no_scaled[b][h] << std::endl;
	// 	}
	// }
	// std::cout << "output_no_scaled(e^(S-m) * V):" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << att.output_no_scaled[b][h] << std::endl;
	// 	}
	// }


	// 
	for(int b = 0 ; b < B ; ++b){
		auto outputs_vec_map = Eigen::Map<MatfRow>(outputs_vec.data() + b * T * C,T,C);
		EXPECT_TRUE(outputs_vec_map.isApprox(outputs_res[b], 0.01f)) 
			<< "---gpu---\n"<<outputs_vec_map.block<3,3>(0,0) << std::endl 
			<< " ---cpu--- \n" << outputs_res[b].block<3,3>(0,0) <<std::endl ;
	}

}



// multi T 
TEST(CudaAttention,flash_attention_f32_forward3){
	constexpr size_t B = 4;
	constexpr size_t T = 64;
    constexpr size_t NH = 12 ;
    constexpr size_t C = 32 * NH;
    constexpr size_t C3 = 3 * C;

    auto att =  TestAttention(B,T,C,NH);

    VecBTC inputs(B);
	StdVec<float> inputs_vec(B*T*C3);
	for(int i =0 ; i < B ; ++i){
		inputs[i] = Matf::Random(T,C3);
        Eigen::Map<MatfRow> (inputs_vec.data() + i * T*C3 , T,C3)  = inputs[i];
	}



	VecBTC outputs_res ;
	{
		auto start = std::chrono::high_resolution_clock::now();
		outputs_res = att.ForwardWithNoCausal(inputs);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}


    StdVec<float> outputs_vec(B*T*C);

	{
		auto start = std::chrono::high_resolution_clock::now();
        gpt2cuda::BatchAttentionForward(outputs_vec.data(), inputs_vec.data(), B, T, C3,NH);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}


	// std::cout << "outputs_vec : " << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	auto outputs_vec_map = Eigen::Map<MatfRow>(outputs_vec.data() + b * T * C,T,C);
	// 	std::cout << outputs_vec_map << std::endl;
	// 	std::cout << "---------------\n";
	// }

	// std::cout << "Q:" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << inputs[b].block(0,0,T,C).block(0,h* C/NH,T,C/NH) << std::endl;
	// 	}
	// }

	// std::cout << "K:" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << inputs[b].block(0,C,T,C).block(0,h*C/NH,T,C/NH) << std::endl;
	// 	}
	// }

	// std::cout << "V:" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << inputs[b].block(0,2*C,T,C).block(0,h*C/NH,T,C/NH) << std::endl;
	// 	}
	// }

	// std::cout << "outputs_res : " << B << ", " << T  << ", " << C  << std::endl;
	// for(auto & v : outputs_res){
	// 	std::cout << v << std::endl;
	// }


	// std::cout << "pre_att_no_scaled(S):" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << att.pre_att_no_scaled[b][h] << std::endl;
	// 	}
	// }
	// std::cout << "att_no_scaled(e^(S-m)):" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << att.att_no_scaled[b][h] << std::endl;
	// 	}
	// }
	// std::cout << "output_no_scaled(e^(S-m) * V):" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << att.output_no_scaled[b][h] << std::endl;
	// 	}
	// }


	// 
	for(int b = 0 ; b < B ; ++b){
		auto outputs_vec_map = Eigen::Map<MatfRow>(outputs_vec.data() + b * T * C,T,C);
		EXPECT_TRUE(outputs_vec_map.isApprox(outputs_res[b], 0.01f)) 
			<< "---gpu---\n"<<outputs_vec_map.block<3,3>(0,0) << std::endl 
			<< " ---cpu--- \n" << outputs_res[b].block<3,3>(0,0) <<std::endl ;
	}

}



// multi T with C / NH = 64 
TEST(CudaAttention,flash_attention_f32_forward4){
	constexpr size_t B = 4;
	constexpr size_t T = 64;
    constexpr size_t NH = 12;
    constexpr size_t C = 64 * NH;
    constexpr size_t C3 = 3 * C;

    auto att =  TestAttention(B,T,C,NH);

    VecBTC inputs(B);
	StdVec<float> inputs_vec(B*T*C3);
	for(int i =0 ; i < B ; ++i){
		inputs[i] = Matf::Random(T,C3);
        Eigen::Map<MatfRow> (inputs_vec.data() + i * T*C3 , T,C3)  = inputs[i];
	}



	VecBTC outputs_res ;
	{
		auto start = std::chrono::high_resolution_clock::now();
		outputs_res = att.ForwardWithNoCausal(inputs);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}


    StdVec<float> outputs_vec(B*T*C);

	{
		auto start = std::chrono::high_resolution_clock::now();
        gpt2cuda::BatchAttentionForward(outputs_vec.data(), inputs_vec.data(), B, T, C3,NH);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}


	// std::cout << "outputs_vec : " << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	auto outputs_vec_map = Eigen::Map<MatfRow>(outputs_vec.data() + b * T * C,T,C);
	// 	std::cout << outputs_vec_map << std::endl;
	// 	std::cout << "---------------\n";
	// }

	// std::cout << "Q:" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << inputs[b].block(0,0,T,C).block(0,h* C/NH,T,C/NH) << std::endl;
	// 	}
	// }

	// std::cout << "K:" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << inputs[b].block(0,C,T,C).block(0,h*C/NH,T,C/NH) << std::endl;
	// 	}
	// }

	// std::cout << "V:" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << inputs[b].block(0,2*C,T,C).block(0,h*C/NH,T,C/NH) << std::endl;
	// 	}
	// }

	// std::cout << "outputs_res : " << B << ", " << T  << ", " << C  << std::endl;
	// for(auto & v : outputs_res){
	// 	std::cout << v << std::endl;
	// }


	// std::cout << "pre_att_no_scaled(S):" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << att.pre_att_no_scaled[b][h] << std::endl;
	// 	}
	// }
	// std::cout << "att_no_scaled(e^(S-m)):" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << att.att_no_scaled[b][h] << std::endl;
	// 	}
	// }
	// std::cout << "output_no_scaled(e^(S-m) * V):" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << att.output_no_scaled[b][h] << std::endl;
	// 	}
	// }


	// 
	for(int b = 0 ; b < B ; ++b){
		auto outputs_vec_map = Eigen::Map<MatfRow>(outputs_vec.data() + b * T * C,T,C);
		EXPECT_TRUE(outputs_vec_map.isApprox(outputs_res[b], 0.001f)) 
			<< "---gpu---\n"<<outputs_vec_map.block<3,3>(0,0) << std::endl 
			<< " ---cpu--- \n" << outputs_res[b].block<3,3>(0,0) <<std::endl ;
	}

}

// min attention with casual example
TEST(CudaAttention,flash_attention_casual_f32_forward1){
	constexpr size_t B = 1;
	constexpr size_t T = 64;
    constexpr size_t NH = 1 ;
    constexpr size_t C = 32 * NH;
    constexpr size_t C3 = 3 * C;

    auto att =  Attention(B,T,C,NH);

    VecBTC inputs(B);
	StdVec<float> inputs_vec(B*T*C3);
	for(int i =0 ; i < B ; ++i){
		inputs[i] = Matf::Random(T,C3);
        Eigen::Map<MatfRow> (inputs_vec.data() + i * T*C3 , T,C3)  = inputs[i];
	}



	VecBTC outputs_res ;
	{
		auto start = std::chrono::high_resolution_clock::now();
		outputs_res = att.Forward(inputs);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}


    StdVec<float> outputs_vec(B*T*C);

	{
		auto start = std::chrono::high_resolution_clock::now();
        gpt2cuda::BatchCausalAttentionForward(outputs_vec.data(), inputs_vec.data(), B, T, C3,NH);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}


	for(int b = 0 ; b < B ; ++b){
		auto outputs_vec_map = Eigen::Map<MatfRow>(outputs_vec.data() + b * T * C,T,C);
		EXPECT_TRUE(outputs_vec_map.isApprox(outputs_res[b], 0.01f)) 
			<< "---gpu---\n"<<outputs_vec_map.block<3,3>(0,0) << std::endl 
			<< " ---cpu--- \n" << outputs_res[b].block<3,3>(0,0) <<std::endl ;
	}

}




// min attention with casual example
TEST(CudaAttention,flash_attention_casual_f32_forward2){
	constexpr size_t B = 16;
	constexpr size_t T = 256;
    constexpr size_t NH = 12;
    constexpr size_t C = 64 * NH;
    constexpr size_t C3 = 3 * C;

    auto att =  Attention(B,T,C,NH);

    VecBTC inputs(B);
	StdVec<float> inputs_vec(B*T*C3);
	for(int i =0 ; i < B ; ++i){
		inputs[i] = Matf::Random(T,C3);
        Eigen::Map<MatfRow> (inputs_vec.data() + i * T*C3 , T,C3)  = inputs[i];
	}



	VecBTC outputs_res ;
	{
		auto start = std::chrono::high_resolution_clock::now();
		outputs_res = att.Forward(inputs);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}


    StdVec<float> outputs_vec(B*T*C);

	{
		auto start = std::chrono::high_resolution_clock::now();
        gpt2cuda::BatchCausalAttentionForward(outputs_vec.data(), inputs_vec.data(), B, T, C3,NH);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}

	// std::cout << "outputs_res : " << B << ", " << T  << ", " << C  << std::endl;
	// for(auto & v : outputs_res){
	// 	std::cout << v << std::endl;
	// }

	// std::cout << "outputs_vec : " << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	auto outputs_vec_map = Eigen::Map<MatfRow>(outputs_vec.data() + b * T * C,T,C);
	// 	std::cout << outputs_vec_map << std::endl;
	// 	std::cout << "---------------\n";
	// }

	for(int b = 0 ; b < B ; ++b){
		auto outputs_vec_map = Eigen::Map<MatfRow>(outputs_vec.data() + b * T * C,T,C);
		EXPECT_TRUE(outputs_vec_map.isApprox(outputs_res[b], 0.01f)) 
			<< "---gpu---\n"<<outputs_vec_map.block<3,3>(0,0) << std::endl 
			<< " ---cpu--- \n" << outputs_res[b].block<3,3>(0,0) <<std::endl ;
	}

}