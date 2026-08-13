#include "cuda/gelu.cuh"
#include "matmul.cuh"
#include "residual.cuh"
#include "layer.cuh"
#include "layernorm.cuh"
#include "attention.cuh"
#include <fstream>

namespace gpt2cuda{


    void Layer::Init(size_t B, size_t T, size_t C, size_t V, size_t NH){
        params_bytes_ = 0;
        // 把这个函数内的所有的makepinvec<float>全部换成makePinVecf,所有类似的写法都换
        l_ln1_ = gpt2cuda::makePinVecf({B,T,C});
        l_ln1_means_ = gpt2cuda::makePinVecf({B,T});
        l_ln1_rstds_ = gpt2cuda::makePinVecf({B,T});
        l_ln1_gamma = gpt2cuda::makePinVecf({C});
        l_ln1_beta = gpt2cuda::makePinVecf({C});
        params_bytes_ += l_ln1_.size() * sizeof(float);
        params_bytes_ += l_ln1_means_.size() * sizeof(float);
        params_bytes_ += l_ln1_rstds_.size() * sizeof(float);
        params_bytes_ += l_ln1_gamma.size() * sizeof(float);
        params_bytes_ += l_ln1_beta.size() * sizeof(float);

        l_qkv_ = gpt2cuda::makePinVecf({B,T,3*C});

        l_qkv_weight = gpt2cuda::makePinVecf({C,3*C});
        l_qkv_bias = gpt2cuda::makePinVecf({3*C});
        params_bytes_ += l_qkv_.size() * sizeof(float);
        params_bytes_ += l_qkv_weight.size() * sizeof(float);
        params_bytes_ += l_qkv_bias.size() * sizeof(float);

        l_atty_ = gpt2cuda::makePinVecf({B,T,C});
        l_logsumexp_ = gpt2cuda::makePinVecf({B,NH,T});
        params_bytes_ += l_atty_.size() * sizeof(float);
        params_bytes_ += l_logsumexp_.size() * sizeof(float);

    
        l_attproj_ = gpt2cuda::makePinVecf({B,T,C});
        l_attproj_weight = gpt2cuda::makePinVecf({C,C});
        l_attproj_bias = gpt2cuda::makePinVecf({C});
        params_bytes_ += l_attproj_.size() * sizeof(float);
        params_bytes_ += l_attproj_weight.size() * sizeof(float);
        params_bytes_ += l_attproj_bias.size() * sizeof(float);

        l_residual2_ = gpt2cuda::makePinVecf({B,T,C});
        params_bytes_ += l_residual2_.size() * sizeof(float);




        l_fch_ = gpt2cuda::makePinVecf({B,T,4*C});
        l_fch_gelu_ = gpt2cuda::makePinVecf({B,T,4*C});
        l_fch_weight   = gpt2cuda::makePinVecf({C,4*C});
        l_fch_bias     = gpt2cuda::makePinVecf({4*C});
        params_bytes_ += l_fch_.size() * sizeof(float);
        params_bytes_ += l_fch_gelu_.size() * sizeof(float);
        params_bytes_ += l_fch_weight.size() * sizeof(float);
        params_bytes_ += l_fch_bias.size() * sizeof(float);

        l_ln2_ = gpt2cuda::makePinVecf({B,T,C});
        l_ln2_means_    = gpt2cuda::makePinVecf({B,T});
        l_ln2_rstds_    = gpt2cuda::makePinVecf({B,T});
        l_ln2_gamma    = gpt2cuda::makePinVecf({C});
        l_ln2_beta     = gpt2cuda::makePinVecf({C});
        params_bytes_ += l_ln2_.size() * sizeof(float);
        params_bytes_ += l_ln2_means_.size() * sizeof(float);
        params_bytes_ += l_ln2_rstds_.size() * sizeof(float);
        params_bytes_ += l_ln2_gamma.size() * sizeof(float);
        params_bytes_ += l_ln2_beta.size() * sizeof(float);



        l_fcproj_        = gpt2cuda::makePinVecf({B,T,C});
        l_fcproj_weight = gpt2cuda::makePinVecf({4*C,C});
        l_fcproj_bias  = gpt2cuda::makePinVecf({C});

        params_bytes_ += l_fcproj_.size() * sizeof(float);
        params_bytes_ += l_fcproj_weight.size() * sizeof(float);
        params_bytes_ += l_fcproj_bias.size() * sizeof(float);




        dl_residual2 = gpt2cuda::makePinVecfZero({B, T, C});

        dl_fcproj =gpt2cuda::makePinVecfZero({B, T, C});
        dl_fcproj_weight = gpt2cuda::makePinVecfZero({4*C,C});
        dl_fcproj_bias = gpt2cuda::makePinVecfZero({C});
        params_bytes_ += dl_fcproj.size() * sizeof(float);
        params_bytes_ += dl_fcproj_weight.size() * sizeof(float);
        params_bytes_ += dl_fcproj_bias.size() * sizeof(float);

        dl_fch_gelu =gpt2cuda::makePinVecfZero({B, T, 4 * C});
        params_bytes_ += dl_fch_gelu.size() * sizeof(float);

        dl_fch =gpt2cuda::makePinVecfZero({B, T, 4 * C});
        dl_fch_weight = gpt2cuda::makePinVecfZero({C,4*C});
        dl_fch_bias = gpt2cuda::makePinVecfZero({4*C});
        params_bytes_ += dl_fch.size() * sizeof(float);
        params_bytes_ += dl_fch_weight.size() * sizeof(float);
        params_bytes_ += dl_fch_bias.size() * sizeof(float);


        dl_ln2 =gpt2cuda::makePinVecfZero({B, T, C});
        dl_ln2_gamma = gpt2cuda::makePinVecfZero({C});
        dl_ln2_beta = gpt2cuda::makePinVecfZero({C});
        params_bytes_ += dl_ln2.size() * sizeof(float);
        params_bytes_ += dl_ln2_gamma.size() * sizeof(float);
        params_bytes_ += dl_ln2_beta.size() * sizeof(float);


        dl_attproj =gpt2cuda::makePinVecfZero({B, T, C});
        dl_attproj_weight = gpt2cuda::makePinVecfZero({C,C});
        dl_attproj_bias =    gpt2cuda::makePinVecfZero({C});
        params_bytes_ += dl_attproj.size() * sizeof(float);
        params_bytes_ += dl_attproj_weight.size() * sizeof(float);
        params_bytes_ += dl_attproj_bias.size() * sizeof(float);

        dl_atty =gpt2cuda::makePinVecfZero({B, T, C});
        params_bytes_ += dl_atty.size() * sizeof(float);


        dl_qkv =gpt2cuda::makePinVecfZero({B, T, 3 * C});
        dl_qkv_weight = gpt2cuda::makePinVecfZero({C,3*C});    //(C,3*C)
        dl_qkv_bias = gpt2cuda::makePinVecfZero({3*C});      //(3*C)
        params_bytes_ += dl_qkv.size() * sizeof(float);
        params_bytes_ += dl_qkv_weight.size() * sizeof(float);
        params_bytes_ += dl_qkv_bias.size() * sizeof(float);

        dl_ln1 =gpt2cuda::makePinVecfZero({B, T, C});
        dl_ln1_gamma = gpt2cuda::makePinVecfZero({C}); 
        dl_ln1_beta =  gpt2cuda::makePinVecfZero({C});
        params_bytes_ += dl_ln1.size() * sizeof(float);
        params_bytes_ += dl_ln1_gamma.size() * sizeof(float);
        params_bytes_ += dl_ln1_beta.size() * sizeof(float);
    }

    void Layer::Forward(PinVecf& residual3 , const PinVecf &residual){


        BatchLayerNormForward(l_ln1_.data(), l_ln1_means_.data(), l_ln1_rstds_.data(), residual.data(), l_ln1_gamma.data(),l_ln1_beta.data(),B_,T_,C_);

        BatchMatmulNTForward(l_qkv_.data(), l_ln1_.data(), l_qkv_weight.data(), l_qkv_bias.data(), B_,T_,C_,3*C_);

        BatchCausalAttentionForward(l_atty_.data(),l_logsumexp_.data(),l_qkv_.data(),B_,T_,3 * C_,NH_);

        BatchMatmulNTForward(l_attproj_.data(),l_atty_.data(),l_attproj_weight.data(),l_attproj_bias.data(),B_,T_,C_,C_); 

        BatchResidualForward(l_residual2_.data(),residual.data(),l_attproj_.data(),B_,T_,C_);

        BatchLayerNormForward(l_ln2_.data(), l_ln2_means_.data(), l_ln2_rstds_.data(), l_residual2_.data(), l_ln2_gamma.data(), l_ln2_beta.data(), B_,T_,C_);
        
        BatchMatmulNTForward(l_fch_.data(),l_ln2_.data(),l_fch_weight.data(),l_fch_bias.data(),B_,T_,C_,4*C_);

        BatchGeluForward(l_fch_gelu_.data(),l_fch_.data(),B_,T_,4 * C_);

        // if reference
        // BatchMatmulGeluForward(l_fch_gelu_.data(),l_ln2_.data(),l_fch_weight_.data(),l_fch_bias_.data(),B_,T_,C_,4*C_);

        BatchMatmulNTForward(l_fcproj_.data(),l_fch_gelu_.data(),l_fcproj_weight.data(),l_fcproj_bias.data(),B_,T_,4*C_,C_);

        BatchResidualForward(residual3.data(),l_residual2_.data(),l_fcproj_.data(),B_,T_,C_);

    }

    void gpt2cuda::Layer::Backward(PinVecf& dresidual , const PinVecf&d_outputs, const PinVecf &residual){
        // std::ofstream cuda_log {"cuda_layer_log.txt" ,std::ios_base::out | std::ios_base::trunc};
        // cuda_log<< "-----------cuda layer backward-----------" << std::endl;
        // #define quick_debug_print(vec) cuda_log<< __LINE__ <<  " line  "#vec" :"  << vec[0] << "," << vec[1] << std::endl
        #define quick_debug_print(vec) 

        // quick_debug_print(d_outputs);
        // quick_debug_print(residual);
        //second residual

        quick_debug_print(dl_residual2);
        quick_debug_print(dl_fcproj);
        BatchResidualBackward(dl_residual2.data(), dl_fcproj.data(), d_outputs.data(), B_, T_, C_);
        // quick_debug_print(dl_residual2);
        // quick_debug_print(dl_fcproj);
        quick_debug_print(dl_fch_gelu);
        quick_debug_print(dl_fch_weight);
        quick_debug_print(dl_fch_bias);
        BatchMatmulNTBackward(dl_fch_gelu.data(),dl_fcproj_weight.data(),dl_fcproj_bias.data(),dl_fcproj.data(),l_fch_gelu_.data(),l_fcproj_weight.data(),B_,T_,4*C_,C_);

        // quick_debug_print(dl_fch_gelu);
        // quick_debug_print(dl_fch_weight);
        // quick_debug_print(dl_fch_bias);

        quick_debug_print(dl_fch);
        BatchGeluBackward(dl_fch.data(), dl_fch_gelu.data(), l_fch_.data(),B_,T_,4*C_);
        // quick_debug_print(dl_fch);
        // quick_debug_print(dl_fch_gelu);
        quick_debug_print(dl_ln2);
        quick_debug_print(dl_fch_weight);
        quick_debug_print(dl_fch_bias);
        BatchMatmulNTBackward(dl_ln2.data(),dl_fch_weight.data(),dl_fch_bias.data(),dl_fch.data(),l_ln2_.data(),l_fch_weight.data(),B_,T_,C_,4*C_);
        // quick_debug_print(dl_ln2);
        // quick_debug_print(dl_fch_weight);

        quick_debug_print(dl_residual2);
        quick_debug_print(dl_ln2_gamma);
        quick_debug_print(dl_ln2_beta);
        BatchLayerNormBackward(dl_residual2.data(), dl_ln2_gamma.data(), dl_ln2_beta.data(), dl_ln2.data(), l_residual2_.data(), l_ln2_gamma.data(),l_ln2_means_.data(), l_ln2_rstds_.data(), B_, T_, C_);
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
        BatchResidualBackward(dresidual.data(), dl_attproj.data(), dl_residual2.data(),B_,T_,C_);

        quick_debug_print(dl_atty);
        quick_debug_print(dl_attproj_weight);
        quick_debug_print(dl_attproj_bias);
        BatchMatmulNTBackward(dl_atty.data(), dl_attproj_weight.data(),dl_attproj_bias.data(),dl_attproj.data(),l_atty_.data(),l_attproj_weight.data(),B_,T_,C_,C_);


        quick_debug_print(dl_qkv);
        BatchCausalAttentionBackward(dl_qkv.data(),dl_atty.data(),l_atty_.data(),l_qkv_.data(),l_logsumexp_.data(),B_,T_,3*C_,NH_);
        // quick_debug_print(dl_qkv);
        // quick_debug_print(dl_atty);
        // quick_debug_print(l_atty_);
        // quick_debug_print(l_qkv_);
        // quick_debug_print(l_logsumexp_);


        quick_debug_print(dl_ln1);
        quick_debug_print(dl_qkv_weight);
        quick_debug_print(dl_qkv_bias);
        BatchMatmulNTBackward(dl_ln1.data(), dl_qkv_weight.data(), dl_qkv_bias.data(),dl_qkv.data(),l_ln1_.data(),l_qkv_weight.data(),B_,T_,C_,3*C_);
        // quick_debug_print(dl_ln1);
        // quick_debug_print(dl_qkv_weight);
        // quick_debug_print(dl_qkv_bias);

        quick_debug_print(dresidual);
        quick_debug_print(dl_ln1_gamma);
        quick_debug_print(dl_ln1_beta);
        BatchLayerNormBackward(dresidual.data(),dl_ln1_gamma.data(),dl_ln1_beta.data(),dl_ln1.data(),residual.data(),l_ln1_gamma.data(),l_ln1_means_.data(),l_ln1_rstds_.data(),B_,T_,C_);
        // quick_debug_print(dresidual);
        // quick_debug_print(dl_ln1_gamma);
        // quick_debug_print(dl_ln1_beta);

        #undef quick_debug_print
    }
}
