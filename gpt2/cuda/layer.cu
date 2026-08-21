#include "cuda/gelu.cuh"
#include "matmul.cuh"
#include "residual.cuh"
#include "layer.cuh"
#include "layernorm.cuh"
#include "attention.cuh"
#include <fstream>
#include "devvector.cuh"
#include <nvtx3/nvtx3.hpp>

namespace gpt2cuda{


    void Layer::Init(size_t B, size_t T, size_t C, size_t V, size_t NH){
        params_bytes_ = 0;
        

        l_ln1_ = makeDevVecf(B*T*C);
        l_ln1_means_ = makeDevVecf(B*T);
        l_ln1_rstds_ = makeDevVecf(B*T);
        l_ln1_gamma = makeDevVecf(C);
        l_ln1_beta = makeDevVecf(C);
        params_bytes_ += l_ln1_.size() * sizeof(float);
        params_bytes_ += l_ln1_means_.size() * sizeof(float);
        params_bytes_ += l_ln1_rstds_.size() * sizeof(float);
        params_bytes_ += l_ln1_gamma.size() * sizeof(float);
        params_bytes_ += l_ln1_beta.size() * sizeof(float);

        l_qkv_ = gpt2cuda::makeDevVecf(B*T*3*C);

        l_qkv_weight = gpt2cuda::makeDevVecf(C*3*C);
        l_qkv_bias = gpt2cuda::makeDevVecf(3*C);
        params_bytes_ += l_qkv_.size() * sizeof(float);
        params_bytes_ += l_qkv_weight.size() * sizeof(float);
        params_bytes_ += l_qkv_bias.size() * sizeof(float);

        l_atty_ = gpt2cuda::makeDevVecf(B*T*C);
        l_logsumexp_ = gpt2cuda::makeDevVecf(B*NH*T);
        params_bytes_ += l_atty_.size() * sizeof(float);
        params_bytes_ += l_logsumexp_.size() * sizeof(float);

    
        l_attproj_ = gpt2cuda::makeDevVecf(B*T*C);
        l_attproj_weight = gpt2cuda::makeDevVecf(C*C);
        l_attproj_bias = gpt2cuda::makeDevVecf(C);
        params_bytes_ += l_attproj_.size() * sizeof(float);
        params_bytes_ += l_attproj_weight.size() * sizeof(float);
        params_bytes_ += l_attproj_bias.size() * sizeof(float);


        l_residual2_ = gpt2cuda::makeDevVecf(B*T*C);
        params_bytes_ += l_residual2_.size() * sizeof(float);





        l_fch_ = gpt2cuda::makeDevVecf(B*T*4*C);
        l_fch_gelu_ = gpt2cuda::makeDevVecf(B*T*4*C);
        l_fch_weight   = gpt2cuda::makeDevVecf(C*4*C);
        l_fch_bias     = gpt2cuda::makeDevVecf(4*C);
        params_bytes_ += l_fch_.size() * sizeof(float);
        params_bytes_ += l_fch_gelu_.size() * sizeof(float);
        params_bytes_ += l_fch_weight.size() * sizeof(float);
        params_bytes_ += l_fch_bias.size() * sizeof(float);

        l_ln2_ = gpt2cuda::makeDevVecf(B*T*C);
        l_ln2_means_    = gpt2cuda::makeDevVecf(B*T);
        l_ln2_rstds_    = gpt2cuda::makeDevVecf(B*T);
        l_ln2_gamma    = gpt2cuda::makeDevVecf(C);
        l_ln2_beta     = gpt2cuda::makeDevVecf(C);
        params_bytes_ += l_ln2_.size() * sizeof(float);
        params_bytes_ += l_ln2_means_.size() * sizeof(float);
        params_bytes_ += l_ln2_rstds_.size() * sizeof(float);
        params_bytes_ += l_ln2_gamma.size() * sizeof(float);
        params_bytes_ += l_ln2_beta.size() * sizeof(float);


        // 所有的变量都替换成 makeDevVecf ，构造函数改为 x*x*x


        l_fcproj_        = gpt2cuda::makeDevVecf(B*T*C);
        l_fcproj_weight = gpt2cuda::makeDevVecf(4*C*C);
        l_fcproj_bias  = gpt2cuda::makeDevVecf(C);

        params_bytes_ += l_fcproj_.size() * sizeof(float);
        params_bytes_ += l_fcproj_weight.size() * sizeof(float);
        params_bytes_ += l_fcproj_bias.size() * sizeof(float);





        dl_residual2 = gpt2cuda::makeDevVecfZero(B* T* C);

        dl_fcproj =gpt2cuda::makeDevVecfZero(B* T* C);
        dl_fcproj_weight = gpt2cuda::makeDevVecfZero(4*C*C);
        dl_fcproj_bias = gpt2cuda::makeDevVecfZero(C);
        params_bytes_ += dl_fcproj.size() * sizeof(float);
        params_bytes_ += dl_fcproj_weight.size() * sizeof(float);
        params_bytes_ += dl_fcproj_bias.size() * sizeof(float);

        dl_fch_gelu =gpt2cuda::makeDevVecfZero(B* T* 4 * C);
        params_bytes_ += dl_fch_gelu.size() * sizeof(float);

        dl_fch =gpt2cuda::makeDevVecfZero(B* T* 4 * C);
        dl_fch_weight = gpt2cuda::makeDevVecfZero(C*4*C);
        dl_fch_bias = gpt2cuda::makeDevVecfZero(4*C);
        params_bytes_ += dl_fch.size() * sizeof(float);
        params_bytes_ += dl_fch_weight.size() * sizeof(float);
        params_bytes_ += dl_fch_bias.size() * sizeof(float);



        dl_ln2 =gpt2cuda::makeDevVecfZero(B* T* C);
        dl_ln2_gamma = gpt2cuda::makeDevVecfZero(C);
        dl_ln2_beta = gpt2cuda::makeDevVecfZero(C);
        params_bytes_ += dl_ln2.size() * sizeof(float);
        params_bytes_ += dl_ln2_gamma.size() * sizeof(float);
        params_bytes_ += dl_ln2_beta.size() * sizeof(float);


        dl_attproj =gpt2cuda::makeDevVecfZero(B* T* C);
        dl_attproj_weight = gpt2cuda::makeDevVecfZero(C*C);
        dl_attproj_bias =    gpt2cuda::makeDevVecfZero(C);
        params_bytes_ += dl_attproj.size() * sizeof(float);
        params_bytes_ += dl_attproj_weight.size() * sizeof(float);
        params_bytes_ += dl_attproj_bias.size() * sizeof(float);

        dl_atty =gpt2cuda::makeDevVecfZero(B* T* C);
        params_bytes_ += dl_atty.size() * sizeof(float);
        att_D = gpt2cuda::makeDevVecf(B*NH*T);


        dl_qkv =gpt2cuda::makeDevVecfZero(B* T* 3 * C);
        dl_qkv_weight = gpt2cuda::makeDevVecfZero(C*3*C);    //(C,3*C)
        dl_qkv_bias = gpt2cuda::makeDevVecfZero(3*C);      //(3*C)
        params_bytes_ += dl_qkv.size() * sizeof(float);
        params_bytes_ += dl_qkv_weight.size() * sizeof(float);
        params_bytes_ += dl_qkv_bias.size() * sizeof(float);

        dl_ln1 =gpt2cuda::makeDevVecfZero(B* T* C);
        dl_ln1_gamma = gpt2cuda::makeDevVecfZero(C); 
        dl_ln1_beta =  gpt2cuda::makeDevVecfZero(C);
        params_bytes_ += dl_ln1.size() * sizeof(float);
        params_bytes_ += dl_ln1_gamma.size() * sizeof(float);
        params_bytes_ += dl_ln1_beta.size() * sizeof(float);
    }

	static const nvtx3::registered_string tag_fwd{"layer forward"};
    void Layer::Forward(DevVecf& residual3 , const DevVecf &residual,cudaStream_t stream){
        nvtx3::scoped_range nvtx_fwd{tag_fwd};

        BatchLayerNormForward(l_ln1_, l_ln1_means_, l_ln1_rstds_, residual, l_ln1_gamma,l_ln1_beta,B_,T_,C_,stream);

        BatchMatmulNTForward(l_qkv_, l_ln1_, l_qkv_weight, l_qkv_bias, B_,T_,C_,3*C_,stream);

        BatchCausalAttentionForward(l_atty_,l_logsumexp_,l_qkv_,B_,T_,3 * C_,NH_,stream);

        BatchMatmulNTForward(l_attproj_,l_atty_,l_attproj_weight,l_attproj_bias,B_,T_,C_,C_,stream); 

        BatchResidualForward(l_residual2_,residual,l_attproj_,B_,T_,C_,stream);

        BatchLayerNormForward(l_ln2_, l_ln2_means_, l_ln2_rstds_, l_residual2_, l_ln2_gamma, l_ln2_beta, B_,T_,C_,stream);
        
        BatchMatmulNTForward(l_fch_,l_ln2_,l_fch_weight,l_fch_bias,B_,T_,C_,4*C_,stream);

        BatchGeluForward(l_fch_gelu_,l_fch_,B_,T_,4 * C_,stream);

        // if reference
        // BatchMatmulGeluForward(l_fch_gelu_.data(),l_ln2_.data(),l_fch_weight_.data(),l_fch_bias_.data(),B_,T_,C_,4*C_);

        BatchMatmulNTForward(l_fcproj_,l_fch_gelu_,l_fcproj_weight,l_fcproj_bias,B_,T_,4*C_,C_,stream);

        BatchResidualForward(residual3,l_residual2_,l_fcproj_,B_,T_,C_,stream);

    }

	static const nvtx3::registered_string tag_bwd{"layer backward"};
    void gpt2cuda::Layer::Backward(DevVecf& dresidual , const DevVecf&d_outputs, const DevVecf &residual,cudaStream_t stream){
        nvtx3::scoped_range nvtx_bwd{tag_bwd};
        // std::ofstream cuda_log {"cuda_layer_log.txt" ,std::ios_base::out | std::ios_base::trunc};
        // cuda_log<< "-----------cuda layer backward-----------" << std::endl;
        // #define quick_debug_print(vec) cuda_log<< __LINE__ <<  " line  "#vec" :"  << vec[0] << "," << vec[1] << std::endl
        #define quick_debug_print(vec) 

        // quick_debug_print(d_outputs);
        // quick_debug_print(residual);
        //second residual

        quick_debug_print(dl_residual2);
        quick_debug_print(dl_fcproj);
        BatchResidualBackward(dl_residual2, dl_fcproj, d_outputs, B_, T_, C_,stream);
        // quick_debug_print(dl_residual2);
        // quick_debug_print(dl_fcproj);
        quick_debug_print(dl_fch_gelu);
        quick_debug_print(dl_fch_weight);
        quick_debug_print(dl_fch_bias);
        BatchMatmulNTBackward(dl_fch_gelu,dl_fcproj_weight,dl_fcproj_bias,dl_fcproj,l_fch_gelu_,l_fcproj_weight,B_,T_,4*C_,C_,stream);

        // quick_debug_print(dl_fch_gelu);
        // quick_debug_print(dl_fch_weight);
        // quick_debug_print(dl_fch_bias);

        quick_debug_print(dl_fch);
        BatchGeluBackward(dl_fch, dl_fch_gelu, l_fch_,B_,T_,4*C_,stream);
        // quick_debug_print(dl_fch);
        // quick_debug_print(dl_fch_gelu);
        quick_debug_print(dl_ln2);
        quick_debug_print(dl_fch_weight);
        quick_debug_print(dl_fch_bias);
        BatchMatmulNTBackward(dl_ln2,dl_fch_weight,dl_fch_bias,dl_fch,l_ln2_,l_fch_weight,B_,T_,C_,4*C_,stream);
        // quick_debug_print(dl_ln2);
        // quick_debug_print(dl_fch_weight);

        quick_debug_print(dl_residual2);
        quick_debug_print(dl_ln2_gamma);
        quick_debug_print(dl_ln2_beta);
        BatchLayerNormBackward(dl_residual2, dl_ln2_gamma, dl_ln2_beta, dl_ln2, l_residual2_, l_ln2_gamma,l_ln2_means_, l_ln2_rstds_, B_, T_, C_,stream);
        // quick_debug_print(l_residual2_);
        // quick_debug_print(dl_residual2);
        // quick_debug_print(dl_ln2_gamma);
        // quick_debug_print(dl_ln2_beta);
        // quick_debug_print(l_ln2_gamma);
        // quick_debug_print(l_ln2_beta);
        // quick_debug_print(l_ln2_means_);
        // quick_debug_print(l_ln2_rstds_);

        //first residual
        quick_debug_print(dresidual);
        quick_debug_print(dl_attproj);
        BatchResidualBackward(dresidual, dl_attproj, dl_residual2,B_,T_,C_,stream);

        quick_debug_print(dl_atty);
        quick_debug_print(dl_attproj_weight);
        quick_debug_print(dl_attproj_bias);
        BatchMatmulNTBackward(dl_atty, dl_attproj_weight,dl_attproj_bias,dl_attproj,l_atty_,l_attproj_weight,B_,T_,C_,C_,stream);


        quick_debug_print(dl_qkv);
        BatchCausalAttentionBackward(dl_qkv,att_D,dl_atty,l_atty_,l_qkv_,l_logsumexp_,B_,T_,3*C_,NH_,stream);
        // quick_debug_print(dl_qkv);
        // quick_debug_print(dl_atty);
        // quick_debug_print(l_atty_);
        // quick_debug_print(l_qkv_);
        // quick_debug_print(l_logsumexp_);


        quick_debug_print(dl_ln1);
        quick_debug_print(dl_qkv_weight);
        quick_debug_print(dl_qkv_bias);
        BatchMatmulNTBackward(dl_ln1, dl_qkv_weight, dl_qkv_bias,dl_qkv,l_ln1_,l_qkv_weight,B_,T_,C_,3*C_,stream);
        // quick_debug_print(dl_ln1);
        // quick_debug_print(dl_qkv_weight);
        // quick_debug_print(dl_qkv_bias);

        quick_debug_print(dresidual);
        quick_debug_print(dl_ln1_gamma);
        quick_debug_print(dl_ln1_beta);
        BatchLayerNormBackward(dresidual,dl_ln1_gamma,dl_ln1_beta,dl_ln1,residual,l_ln1_gamma,l_ln1_means_,l_ln1_rstds_,B_,T_,C_,stream);
        // quick_debug_print(dresidual);
        // quick_debug_print(dl_ln1_gamma);
        // quick_debug_print(dl_ln1_beta);

        #undef quick_debug_print
    }
}

