#include "cuda/gelu.cuh"
#include "matmul.cuh"
#include "residual.cuh"
#include "layer.cuh"
#include "layernorm.cuh"
#include "attention.cuh"


using namespace gpt2cuda;

void Layer::Init(size_t B, size_t T, size_t C, size_t V, size_t NH){

    l_ln1_ = gpt2cuda::makeVec<float>({B,T,C});
    l_ln1_means_ = gpt2cuda::makeVec<float>({B,T});
    l_ln1_rstds_ = gpt2cuda::makeVec<float>({B,T});
    l_ln1_gamma = gpt2cuda::makeVec<float>({C});
    l_ln1_beta = gpt2cuda::makeVec<float>({C});

    l_qkv_ = gpt2cuda::makeVec<float>({B,T,3*C});
    l_qkv_weight = gpt2cuda::makeVec<float>({C,3*C});
    l_qkv_bias = gpt2cuda::makeVec<float>({3*C});

    l_atty_ = gpt2cuda::makeVec<float>({B,T,C});
    l_logsumexp_ = gpt2cuda::makeVec<float>({B,NH,T});

   
    l_attproj_ = gpt2cuda::makeVec<float>({B,T,C});
    l_attproj_weight = gpt2cuda::makeVec<float>({C,C});
    l_attproj_bias = gpt2cuda::makeVec<float>({C});
    
    l_residual2_ = gpt2cuda::makeVec<float>({B,T,C});

    l_fch_ = gpt2cuda::makeVec<float>({B,T,4*C});
    l_fch_gelu_ = gpt2cuda::makeVec<float>({B,T,4*C});
    l_fch_weight   = gpt2cuda::makeVec<float>({C,4*C});
    l_fch_bias     = gpt2cuda::makeVec<float>({4*C});

    l_ln2_ = gpt2cuda::makeVec<float>({B,T,C});
    l_ln2_means_    = gpt2cuda::makeVec<float>({B,T});
    l_ln2_rstds_    = gpt2cuda::makeVec<float>({B,T});
    l_ln2_gamma    = gpt2cuda::makeVec<float>({C});
    l_ln2_beta     = gpt2cuda::makeVec<float>({C});


    l_fcproj_        = gpt2cuda::makeVec<float>({B,T,C});
    l_fcproj_weight = gpt2cuda::makeVec<float>({4*C,C});
    l_fcproj_bias  = gpt2cuda::makeVec<float>({C});




	dl_residual2 =gpt2cuda::makeZero<float>({B, T, C});

	dl_fcproj =gpt2cuda::makeZero<float>({B, T, C});
    dl_fcproj_weight = gpt2cuda::makeZero<float>({4*C,C});
    dl_fcproj_bias = gpt2cuda::makeZero<float>({C});

	dl_fch_gelu =gpt2cuda::makeZero<float>({B, T, 4 * C});

	dl_fch =gpt2cuda::makeZero<float>({B, T, 4 * C});
    dl_fch_weight = gpt2cuda::makeZero<float>({C,4*C});
    dl_fch_bias = gpt2cuda::makeZero<float>({4*C});

	dl_ln2 =gpt2cuda::makeZero<float>({B, T, C});
    dl_ln2_gamma = gpt2cuda::makeZero<float>({C});
    dl_ln2_beta = gpt2cuda::makeZero<float>({C});


	dl_attproj =gpt2cuda::makeZero<float>({B, T, C});
    dl_attproj_weight = gpt2cuda::makeZero<float>({C,C});
    dl_attproj_bias =    gpt2cuda::makeZero<float>({C});

	dl_atty =gpt2cuda::makeZero<float>({B, T, C});


	dl_qkv =gpt2cuda::makeZero<float>({B, T, 3 * C});
    dl_qkv_weight = gpt2cuda::makeZero<float>({C,3*C});    //(C,3*C)
    dl_qkv_bias = gpt2cuda::makeZero<float>({3*C});      //(3*C)

	dl_ln1 =gpt2cuda::makeZero<float>({B, T, C});
    dl_ln1_gamma = gpt2cuda::makeZero<float>({C}); 
    dl_ln1_beta =  gpt2cuda::makeZero<float>({C});
}

void Layer::Forward(StdVecf& residual3 , const StdVecf &residual){


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

void gpt2cuda::Layer::Backward(StdVecf& dresidual , const StdVecf&d_outputs, const StdVecf &residual){

    //second residual
    BatchResidualBackward(dl_residual2.data(), dl_fcproj.data(), d_outputs.data(), B_, T_, C_);
    
    BatchMatmulNTBackward(dl_fch_gelu.data(),dl_fcproj_weight.data(),dl_fcproj_bias.data(),dl_fcproj.data(),l_fch_gelu_.data(),l_fcproj_weight.data(),B_,T_,4*C_,C_);

    BatchGeluBackward(dl_fch.data(), dl_fch_gelu.data(), l_fch_.data(),B_,T_,4*C_);

    BatchMatmulNTBackward(dl_ln2.data(),dl_fch_weight.data(),dl_fch_bias.data(),dl_fch.data(),l_ln2_.data(),l_fch_weight.data(),B_,T_,C_,4*C_);

    BatchLayerNormBackward(dl_residual2.data(), dl_ln2_gamma.data(), dl_ln2_beta.data(), dl_ln2.data(), l_residual2_.data(), l_ln2_gamma.data(),l_ln2_means_.data(), l_ln2_rstds_.data(), B_, T_, C_);

    //first residual
    BatchResidualBackward(dresidual.data(), dl_attproj.data(), dl_residual2.data(),B_,T_,C_);

    BatchMatmulNTBackward(dl_atty.data(), dl_attproj_weight.data(),dl_attproj_bias.data(),dl_attproj.data(),l_atty_.data(),l_attproj_weight.data(),B_,T_,C_,C_);


    BatchAttentionBackward(dl_qkv.data(),dl_atty.data(),l_atty_.data(),l_qkv_.data(),l_logsumexp_.data(),B_,T_,3*C_,NH_);


    BatchMatmulNTBackward(dl_ln1.data(), dl_qkv_weight.data(), dl_qkv_bias.data(),dl_qkv.data(),l_ln1_.data(),l_qkv_weight.data(),B_,T_,C_,3*C_);

    BatchLayerNormBackward(dresidual.data(),dl_ln1_gamma.data(),dl_ln1_beta.data(),dl_ln1.data(),residual.data(),l_ln1_gamma.data(),l_ln1_means_.data(),l_ln1_rstds_.data(),B_,T_,C_);

}

