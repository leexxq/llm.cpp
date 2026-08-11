#include "attention.cuh"
#include "error.cuh"
#include "cutlass/util/device_memory.h"
#include "kernel/flash_attn/config.cuh"
#include "kernel/flash_attn/flash_fwd_launch.cuh"
#include "kernel/flash_attn/flash_bwd_launch.cuh"
#include <cassert>
#include <cstdio>
namespace gpt2cuda{

	using DAlloc = cutlass::device_memory::allocation<float>;

	void BatchAttentionForward(float *outputs,float * logsumexp , float const *inputs, AttentionType Attention , int B, int T, int C3, int NH) {
		using namespace cute;
		assert(C3 % 3 == 0);
		auto C = C3 / 3;

		DAlloc outputs_d(B * T * C);
		DAlloc inputs_d(B * T * C3);
		DAlloc logsumexp_d(B * NH * T);

		outputs_d.copy_from_host(outputs);
		inputs_d.copy_from_host(inputs);
		

		if(C/NH == 32){
			if(Attention == AttentionType::Default){
				kernel::AttentionForwardCUDA<AttentionType::Default,32>(outputs_d.get(),logsumexp_d.get(), inputs_d.get(), B, T, C3, NH);
			}else if(Attention == AttentionType::Causal){
				kernel::AttentionForwardCUDA<AttentionType::Causal,32>(outputs_d.get(),logsumexp_d.get(), inputs_d.get(), B, T, C3, NH);
			}else{
				std::cerr << "fatal: " << static_cast<int>(Attention) << " not exists!"<< std::endl;
				exit(1);
			}
		}else if(C/NH == 64){
			if(Attention == AttentionType::Default){
				kernel::AttentionForwardCUDA<AttentionType::Default,64>(outputs_d.get(),logsumexp_d.get(), inputs_d.get(), B, T, C3, NH);
			}else if(Attention == AttentionType::Causal){
				kernel::AttentionForwardCUDA<AttentionType::Causal,64>(outputs_d.get(),logsumexp_d.get(), inputs_d.get(), B, T, C3, NH);
			}else{
				std::cerr << "fatal: " << static_cast<int>(Attention) << " not exists!"<< std::endl;
				exit(1);
			}
		}else
		{
			std::cerr << "not supported attention shape D = " << C/NH << std::endl;
			exit(1);
		}

		outputs_d.copy_to_host(outputs);
		logsumexp_d.copy_to_host(logsumexp);
	}

	void BatchAttentionBackward(float *d_inputs, float const *d_outputs,float const* outputs,  float const *inputs,float const* logsumexp,AttentionType Attention , int B, int T, int C3, int NH) {
		using namespace cute;
		assert(C3 % 3 == 0);
		auto C = C3 / 3;

		auto d_inputs_d = DAlloc(B*T*C3);
		auto d_outputs_d = DAlloc(B*T*C);
		auto outputs_d = DAlloc(B*T*C);
		auto inputs_d = DAlloc(B*T*C3);
		auto logsumexp_d = DAlloc(B*NH*T);

		d_outputs_d.copy_from_host(d_outputs);
		outputs_d.copy_from_host(outputs);
		d_inputs_d.copy_from_host(d_inputs);
		inputs_d.copy_from_host(inputs);
		logsumexp_d.copy_from_host(logsumexp);

		if(C/NH == 32){
			if(Attention == AttentionType::Default){
				kernel::AttentionBackwardCUDA<AttentionType::Default,32>(d_inputs_d.get(),d_outputs_d.get(),outputs_d.get(),inputs_d.get(),logsumexp_d.get(),B,T,C3,NH);
			}else if(Attention == AttentionType::Causal){
				kernel::AttentionBackwardCUDA<AttentionType::Causal,32>(d_inputs_d.get(),d_outputs_d.get(),outputs_d.get(),inputs_d.get(),logsumexp_d.get(),B,T,C3,NH);
			}else{
				std::cerr << "fatal: " << static_cast<int>(Attention) << " not exists!"<< std::endl;
				exit(1);
			}
		}else if(C/NH == 64){
			if(Attention == AttentionType::Default){
				kernel::AttentionBackwardCUDA<AttentionType::Default,64>(d_inputs_d.get(),d_outputs_d.get(),outputs_d.get(),inputs_d.get(),logsumexp_d.get(),B,T,C3,NH);
			}else if(Attention == AttentionType::Causal){
				kernel::AttentionBackwardCUDA<AttentionType::Causal,64>(d_inputs_d.get(),d_outputs_d.get(),outputs_d.get(), inputs_d.get(),logsumexp_d.get(),B,T,C3,NH);
			}else{
				std::cerr << "fatal: " << static_cast<int>(Attention) << " not exists!"<< std::endl;
				exit(1);
			}
		}else
		{
			std::cerr << "not supported attention shape D = " << C/NH << std::endl;
			exit(1);
		}

		d_inputs_d.copy_to_host(d_inputs);
		

	}

	void BatchAttentionForward(float *outputs, float* logsumexp ,float const *inputs, int B, int T, int C3, int NH) {
		BatchAttentionForward(outputs,logsumexp,inputs,AttentionType::Default , B,T,C3,NH);
	}

	void BatchAttentionBackward(float *d_inputs,float const *d_outputs,float const* outputs, float const *inputs,float const* logsumexp ,int B, int T, int C3, int NH) {
		BatchAttentionBackward(d_inputs,d_outputs,outputs,inputs,logsumexp,AttentionType::Default,B,T,C3,NH);
	}

	void BatchCausalAttentionBackward(float *d_inputs,float const *d_outputs,float const* outputs, float const *inputs,float const* logsumexp ,int B, int T, int C3, int NH) {
		BatchAttentionBackward(d_inputs,d_outputs,outputs,inputs,logsumexp,AttentionType::Causal,B,T,C3,NH);
	}


	void BatchCausalAttentionForward(float *outputs,float* logsumexp , float const *inputs, int B, int T, int C3, int NH) {
		BatchAttentionForward(outputs,logsumexp,inputs,AttentionType::Causal , B,T,C3,NH);
	}

}