#include "cuda/attention.cuh"
#include "attention.h"
#include "global.h"
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <chrono>
#include <fstream>
#include <ios>
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
		using VecBHT = VecBTC;
		VecBHT logsumexp;
		VecBHT D;
	
	public:
		VecBHTT d_att;
		VecBHTT d_pre_att;
		
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
					logsumexp[b](h,r) = maxval + std::log(expsum);
				}
			}

			output_no_scaled[b][h] = att_no_scaled[b][h] * vv;
			
			output = att[b][h]  * vv;

			return output;
		}
		void ScaledDotAttentionBackward(VecBT3C &d_inputs, const VecBTC &d_outputs, int b, int h, const Matf &qq, const Matf &kk, const Matf &vv , const VecBTC &outputs){
			size_t seq_len = vv.rows();
			size_t head_channels = vv.cols();
			float scale = 1.f / std::sqrt(head_channels);
			size_t channels_3 = d_inputs.front().cols();
			assert(channels_3 % 3 == 0);
			size_t channels = channels_3 / 3;

			using Eigen::Ref;

			const Matf &d_output = d_outputs[b].block(0, h * head_channels, seq_len, head_channels);

			Ref<Matf> d_qq = d_inputs[b].block(0, 0, seq_len, channels).block(0, h * head_channels, seq_len, head_channels);
			Ref<Matf> d_kk = d_inputs[b].block(0, channels, seq_len, channels).block(0, h * head_channels, seq_len, head_channels);
			Ref<Matf> d_vv = d_inputs[b].block(0, 2 * channels, seq_len, channels).block(0, h * head_channels, seq_len, head_channels);
			const Matf &output = outputs[b].block(0, h * head_channels, seq_len, head_channels);

			const Matf &att_bh = att[b][h];

			Matf& d_att_bh = d_att[b][h];
			Matf& d_pre_att_bh = d_pre_att[b][h];


			d_att_bh = d_output * vv.transpose();


			D[b].row(h) = output.cwiseProduct(d_output).rowwise().sum();

			
			for(int r = 0 ; r < D[b].row(h).size() ; ++r){
				d_pre_att_bh.row(r).array() = att_bh.row(r).array() * (d_att_bh.row(r).array()  - D[b](h,r) );
			}

			d_vv = att_bh.transpose() * d_output;

			d_kk = d_pre_att_bh.transpose() * qq * scale;

			d_qq = d_pre_att_bh * kk * scale;
		}
	public:
		TestAttention() {}
		TestAttention(size_t B, size_t T, size_t C, size_t NH) : Attention(B,T,C,NH),
			pre_att_no_scaled(makeZero(B, NH, T, T)),
			att_no_scaled(makeZero(B, NH, T, T)),
			output_no_scaled(makeZero(B, NH, T, T)),
			logsumexp(makeZero(B, NH,T)),
			
			d_att(makeZero(B,NH,T,T)),
			d_pre_att(makeZero(B,NH,T,T)),
			D(makeZero(B,NH,T))
		{
	
		}
		VecBTC ForwardWithNoCausal(const VecBT3C &inputs) {
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
		VecBT3C BackwardWithNoCausal(const VecBTC &d_outputs, const VecBTC &inputs, const VecBTC & outputs){
			auto [B, T, C3] = GetShape(inputs);
			assert(C3 % 3 == 0);
			size_t C = C3 / 3;
			size_t num_heads = att.front().size();
			size_t head_dim = C / num_heads;
			VecBTC d_inputs = makeZero(B, T, C3);
			for (int b = 0; b < B; ++b) {
				//it's Q part
				const Matf &qQ = inputs[b].block(0, 0, T, C);
				const Matf &kK = inputs[b].block(0, C, T, C);
				const Matf &vV = inputs[b].block(0, 2 * C, T, C);

				for (int h = 0; h < num_heads; ++h) {
					const Matf &qq = qQ.block(0, h * head_dim, T, head_dim);
					const Matf &kk = kK.block(0, h * head_dim, T, head_dim);
					const Matf &vv = vV.block(0, h * head_dim, T, head_dim);
					ScaledDotAttentionBackward(d_inputs, d_outputs, b, h, qq, kk, vv,outputs);
				}
			}
			return d_inputs;
		}
};


void basic_attention_forward_test(size_t B,size_t T,size_t NH,size_t C){

	auto C3 = 3*C;
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
    StdVec<float> logsumexp_vec(B*NH*T);
	{
		auto start = std::chrono::high_resolution_clock::now();
		
        gpt2cuda::BatchAttentionForward(outputs_vec.data(),logsumexp_vec.data(), inputs_vec.data(), B, T, C3,NH);
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

	// std::cout << "logsumexp ( m + log(l) )" << std::endl;
	// for(int b = 0 ; b < B ; ++b){
	// 	std::cout << "--------" << b << "-------\n";
	// 	for(int h = 0; h < NH; ++h){
	// 		std::cout << "-------"<< h << "--------\n";
	// 		std::cout << att.logsumexp[b].row(h) << std::endl;
	// 	}
	// }

	// logsumexp test
	for(int b = 0 ; b < B ; ++b){
		auto logsumexp_vec_map = Eigen::Map<MatfRow>(logsumexp_vec.data() + b * NH * T,NH,T);
		EXPECT_TRUE(logsumexp_vec_map.isApprox(att.logsumexp[b], 0.01f)) 
			<< "---gpu---\n"<<logsumexp_vec_map.row(0) << std::endl 
			<< " ---cpu--- \n" << att.logsumexp[b].row(0) <<std::endl ;
	}



	// result test
	for(int b = 0 ; b < B ; ++b){
		auto outputs_vec_map = Eigen::Map<MatfRow>(outputs_vec.data() + b * T * C,T,C);
		EXPECT_TRUE(outputs_vec_map.isApprox(outputs_res[b], 0.01f)) 
			<< "---gpu---\n"<<outputs_vec_map.block<3,3>(0,0) << std::endl 
			<< " ---cpu--- \n" << outputs_res[b].block<3,3>(0,0) <<std::endl ;
	}

}



// min attention example
TEST(CudaAttention,flash_attention_f32_forward1){
	constexpr size_t B = 1;
	constexpr size_t T = 64;
    constexpr size_t NH = 1 ;
    constexpr size_t C = 32 * NH;
	basic_attention_forward_test(B, T, NH, C);
}



// multi batch  and multi head
TEST(CudaAttention,flash_attention_f32_forward2){
	constexpr size_t B = 4;
	constexpr size_t T = 64;
    constexpr size_t NH = 12 ;
    constexpr size_t C = 32 * NH;
    constexpr size_t C3 = 3 * C;
	basic_attention_forward_test(B, T, NH, C);
}



// multi T 
TEST(CudaAttention,flash_attention_f32_forward3){
	constexpr size_t B = 1;
	constexpr size_t T = 128;
    constexpr size_t NH = 1 ;
    constexpr size_t C = 32 * NH;
    constexpr size_t C3 = 3 * C;

	basic_attention_forward_test(B, T, NH, C);

}



// multi T with C / NH = 64 
TEST(CudaAttention,flash_attention_f32_forward4){
	constexpr size_t B = 4;
	constexpr size_t T = 64;
    constexpr size_t NH = 12;
    constexpr size_t C = 64 * NH;
    constexpr size_t C3 = 3 * C;

	basic_attention_forward_test(B, T, NH, C);
}

void basic_casual_attention_forward_test(size_t B,size_t T,size_t NH,size_t C){
	auto C3 = 3 * C;
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

    StdVec<float> logsumexp_vec(B*NH*T);
	{
		auto start = std::chrono::high_resolution_clock::now();
        gpt2cuda::BatchCausalAttentionForward(outputs_vec.data(), logsumexp_vec.data(),inputs_vec.data(), B, T, C3,NH);
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
TEST(CudaAttention,flash_attention_casual_f32_forward1){
	constexpr size_t B = 1;
	constexpr size_t T = 64;
    constexpr size_t NH = 1 ;
    constexpr size_t C = 32 * NH;
    constexpr size_t C3 = 3 * C;
	basic_casual_attention_forward_test(B, T, NH, C);

}




// attention with casual example
TEST(CudaAttention,flash_attention_casual_f32_forward2){
	constexpr size_t B = 2;
	constexpr size_t T = 256;
    constexpr size_t NH = 2;
    constexpr size_t C = 64 * NH;
    constexpr size_t C3 = 3 * C;
	basic_casual_attention_forward_test(B, T, NH, C);
}



// cd ~/llm.cpp/build && ninja gpt2cudatest && ./tests/gpt2cudatest --gtest_filter='*flash_attention_f32_backward1' > log.txt "/root/llm.cpp/tests/gpt2/cuda/test_cuda_attention.cpp"
// min attention backward example
TEST(CudaAttention,flash_attention_f32_backward1){
	constexpr size_t B = 1;
	constexpr size_t T = 256;
    constexpr size_t NH = 1 ;
    constexpr size_t Hc = 64 ;
    constexpr size_t C = Hc * NH;
    constexpr size_t C3 = 3 * C;

    VecBTC inputs(B);
    VecBTC d_outputs(B);
    auto att =  TestAttention(B,T,C,NH);


    StdVec<float> outputs_vec(B*T*C);
	StdVec<float> d_outputs_vec(B*T*C);

    StdVec<float> logsumexp_vec(B*NH*T);

	StdVec<float> d_inputs_vec(B*T*C3);
	StdVec<float> inputs_vec(B*T*C3);

	for(int i =0 ; i < B ; ++i){
		inputs[i] = Matf::Random(T,C3);
        Eigen::Map<MatfRow> (inputs_vec.data() + i * T*C3 , T,C3)  = inputs[i];
	}

	for(int i =0 ; i < B ; ++i){
		d_outputs[i] = Matf::Random(T,C);
        Eigen::Map<MatfRow> (d_outputs_vec.data() + i * T*C , T,C)  = d_outputs[i];
	}

	VecBTC outputs_res =  att.ForwardWithNoCausal(inputs);
	VecBTC d_inputs_res ;
	{
		auto start = std::chrono::high_resolution_clock::now();
		d_inputs_res = att.BackwardWithNoCausal(d_outputs, inputs, outputs_res);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}


	{
		gpt2cuda::BatchAttentionForward(outputs_vec.data(), logsumexp_vec.data(), inputs_vec.data(), B, T, C3, NH);
		auto start = std::chrono::high_resolution_clock::now();
        gpt2cuda::BatchAttentionBackward(d_inputs_vec.data(),d_outputs_vec.data(),outputs_vec.data(),inputs_vec.data(),logsumexp_vec.data(),B,T,C3,NH);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}


	for (int b = 0; b < B; ++b) {
		Eigen::Map<MatfRow> map_d_intputs_vec(d_inputs_vec.data() + b * T*C3,T,C3);
		EXPECT_TRUE(map_d_intputs_vec.isApprox(d_inputs_res[b],0.01f));
		
	}



	std::ofstream cpu_log{"flash_attention_f32_backward1_cpu_log.txt",std::ios::out | std::ios::trunc};
	std::ofstream gpu_log{"flash_attention_f32_backward1_gpu_log.txt",std::ios::out | std::ios::trunc};
	for (int b = 0; b < B; ++b) {
		//it's Q part
		const Matf &dqQ = d_inputs_res[b].block(0, 0, T, C);
		const Matf &dkK = d_inputs_res[b].block(0, C, T, C);
		const Matf &dvV = d_inputs_res[b].block(0, 2 * C, T, C);
		const Matf &qQ = inputs[b].block(0, 0, T, C);
		const Matf &kK = inputs[b].block(0, C, T, C);
		const Matf &vV = inputs[b].block(0, 2 * C, T, C);

		Eigen::Map<MatfRow> map_d_intputs_vec(d_inputs_vec.data() + b * T*C3,T,C3);
		const Matf &dqQ_d = map_d_intputs_vec.block(0, 0, T, C);
		const Matf &dkK_d = map_d_intputs_vec.block(0, C, T, C);
		const Matf &dvV_d = map_d_intputs_vec.block(0, 2 * C, T, C);


		for (int h = 0; h < NH; ++h) {
			const Matf &dqq = dqQ.block(0, h * Hc, T, Hc);
			const Matf &dkk = dkK.block(0, h * Hc, T, Hc);
			const Matf &dvv = dvV.block(0, h * Hc, T, Hc);

			const Matf &qq = qQ.block(0, h * Hc, T, Hc);
			const Matf &kk = kK.block(0, h * Hc, T, Hc);
			const Matf &vv = vV.block(0, h * Hc, T, Hc);
			const Matf &output = outputs_res[b].block(0, h * Hc, T, Hc);
			const Matf &d_output = d_outputs[b].block(0, h * Hc, T, Hc);

			cpu_log << "Q : \n" << qq << std::endl;
			cpu_log << "D : \n" << att.D[b].row(h) << std::endl;


			cpu_log << "S : " << std::endl;
			cpu_log << att.pre_att[b][h] << std::endl;
			cpu_log << "QK^T(no scaled) : " << std::endl;
			cpu_log << att.pre_att_no_scaled[b][h] << std::endl;
			cpu_log << "QK^T transpose (no scaled) : " << std::endl;
			cpu_log << att.pre_att_no_scaled[b][h].transpose() << std::endl;

			cpu_log << "P : " << std::endl;
			cpu_log << att.att[b][h] << std::endl;

			cpu_log << "dP : " << std::endl;
			cpu_log << att.d_att[b][h] << std::endl;

			cpu_log << "dS : " << std::endl;
			cpu_log << att.d_pre_att[b][h] << std::endl;

			// cpu_log << "V : " << std::endl;
			// cpu_log <<  vv<< std::endl;
			// log << "O : " << std::endl;
			// log <<  output<< std::endl;
			cpu_log << "dO : " << std::endl;
			cpu_log <<  d_output<< std::endl;

			cpu_log << "dQ : " << std::endl;
			cpu_log << dqq << std::endl;

			cpu_log << "dK : " << std::endl;
			cpu_log << dkk << std::endl;

			cpu_log << "dV : " << std::endl;
			cpu_log << dvv << std::endl;



			const Matf &dqq_d = dqQ_d.block(0, h * Hc, T, Hc);
			const Matf &dkk_d = dkK_d.block(0, h * Hc, T, Hc);
			const Matf &dvv_d = dvV_d.block(0, h * Hc, T, Hc);
			gpu_log << "dQ : " << "\n" << dqq_d << std::endl;
			gpu_log << "dK : " << "\n" << dkk_d << std::endl;
			gpu_log << "dV : " << "\n" << dvv_d << std::endl;
		}

	}
}


// cd ~/llm.cpp/build && ninja gpt2cudatest && ./tests/gpt2cudatest --gtest_filter='*flash_causal_attention_f32_backward1' > log.txt "/root/llm.cpp/tests/gpt2/cuda/test_cuda_attention.cpp"
// min attention backward example
TEST(CudaAttention,flash_causal_attention_f32_backward1){
	constexpr size_t B = 4;
	constexpr size_t T = 128;
    constexpr size_t NH = 4 ;
    constexpr size_t Hc = 64 ;
    constexpr size_t C = Hc * NH;
    constexpr size_t C3 = 3 * C;

    VecBTC inputs(B);
    VecBTC d_outputs(B);
    auto att =  TestAttention(B,T,C,NH);


    StdVec<float> outputs_vec(B*T*C);
	StdVec<float> d_outputs_vec(B*T*C);

    StdVec<float> logsumexp_vec(B*NH*T);

	StdVec<float> d_inputs_vec(B*T*C3);
	StdVec<float> inputs_vec(B*T*C3);

	for(int i =0 ; i < B ; ++i){
		inputs[i] = Matf::Random(T,C3);
        Eigen::Map<MatfRow> (inputs_vec.data() + i * T*C3 , T,C3)  = inputs[i];
	}

	for(int i =0 ; i < B ; ++i){
		d_outputs[i] = Matf::Random(T,C);
        Eigen::Map<MatfRow> (d_outputs_vec.data() + i * T*C , T,C)  = d_outputs[i];
	}

	VecBTC outputs_res =  att.Forward(inputs);
	VecBTC d_inputs_res ;
	{
		auto start = std::chrono::high_resolution_clock::now();
		d_inputs_res = att.Backward(d_outputs, inputs);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[cpu]elapsed:" << elapsed << "ms"<<std::endl;
	}


	{
		gpt2cuda::BatchCausalAttentionForward(outputs_vec.data(), logsumexp_vec.data(), inputs_vec.data(), B, T, C3, NH);
		auto start = std::chrono::high_resolution_clock::now();
        gpt2cuda::BatchCausalAttentionBackward(d_inputs_vec.data(),d_outputs_vec.data(),outputs_vec.data(),inputs_vec.data(),logsumexp_vec.data(),B,T,C3,NH);
		auto end= std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[gpu]elapsed:" << elapsed << "ms"<<std::endl;
	}


	for (int b = 0; b < B; ++b) {
		Eigen::Map<MatfRow> map_d_intputs_vec(d_inputs_vec.data() + b * T*C3,T,C3);
		EXPECT_TRUE(map_d_intputs_vec.isApprox(d_inputs_res[b],0.01f));
		
	}


	std::ofstream cpu_log{"flash_causal_attention_f32_backward1_cpu_log.txt",std::ios::out | std::ios::trunc};
	std::ofstream gpu_log{"flash_causal_attention_f32_backward1_gpu_log.txt",std::ios::out | std::ios::trunc};
	for (int b = 0; b < B; ++b) {
		//it's Q part
		const Matf &dqQ = d_inputs_res[b].block(0, 0, T, C);
		const Matf &dkK = d_inputs_res[b].block(0, C, T, C);
		const Matf &dvV = d_inputs_res[b].block(0, 2 * C, T, C);
		const Matf &qQ = inputs[b].block(0, 0, T, C);
		const Matf &kK = inputs[b].block(0, C, T, C);
		const Matf &vV = inputs[b].block(0, 2 * C, T, C);

		Eigen::Map<MatfRow> map_d_intputs_vec(d_inputs_vec.data() + b * T*C3,T,C3);
		const Matf &dqQ_d = map_d_intputs_vec.block(0, 0, T, C);
		const Matf &dkK_d = map_d_intputs_vec.block(0, C, T, C);
		const Matf &dvV_d = map_d_intputs_vec.block(0, 2 * C, T, C);


		for (int h = 0; h < NH; ++h) {
			const Matf &dqq = dqQ.block(0, h * Hc, T, Hc);
			const Matf &dkk = dkK.block(0, h * Hc, T, Hc);
			const Matf &dvv = dvV.block(0, h * Hc, T, Hc);

			const Matf &qq = qQ.block(0, h * Hc, T, Hc);
			const Matf &kk = kK.block(0, h * Hc, T, Hc);
			const Matf &vv = vV.block(0, h * Hc, T, Hc);
			const Matf &output = outputs_res[b].block(0, h * Hc, T, Hc);
			const Matf &d_output = d_outputs[b].block(0, h * Hc, T, Hc);
			const Matf &dqq_d = dqQ_d.block(0, h * Hc, T, Hc);
			const Matf &dkk_d = dkK_d.block(0, h * Hc, T, Hc);
			const Matf &dvv_d = dvV_d.block(0, h * Hc, T, Hc);



			gpu_log << "dQ : " << "\n" << dqq_d << std::endl;
			gpu_log << "dK : " << "\n" << dkk_d << std::endl;
			gpu_log << "dV : " << "\n" << dvv_d << std::endl;

			cpu_log << "D : \n" << output.cwiseProduct(d_output).rowwise().sum() << std::endl;

		}

	}
}