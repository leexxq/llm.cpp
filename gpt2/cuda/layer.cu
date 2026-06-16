#include "matmul.cuh"
#include "residual.cuh"
#include "layer.cuh"
#include "layernorm.cuh"
#include "attention.cuh"


using namespace gpt2cuda;

void Layer::Init(size_t B, size_t T, size_t C, size_t V, size_t NH){


    l_ln1_ = gpt2cuda::makeVecf(B,T,C);
    l_ln1_means_ = gpt2cuda::makeVecf(B,T);
    l_ln1_rstds_ = gpt2cuda::makeVecf(B,T);
    l_ln1_gamma_ = gpt2cuda::makeVecf(C);
    l_ln1_beta_ = gpt2cuda::makeVecf(C);

    l_qkv_ = gpt2cuda::makeVecf(B,T,3*C);
    qkv_weight_ = gpt2cuda::makeVecf(C,3*C);
    qkv_bias_ = gpt2cuda::makeVecf(3*C);

    l_atty_ = gpt2cuda::makeVecf(B,T,C);
    
    l_attproj_ = gpt2cuda::makeVecf(B,T,C);
    l_attproj_weight_ = gpt2cuda::makeVecf(T,C);
    l_attproj_bias_ = gpt2cuda::makeVecf(C);
    
    l_residual2_ = gpt2cuda::makeVecf(B,T,C);

    l_fch_gelu_ = gpt2cuda::makeVecf(B,T,4*C);
    l_fch_weight_   = gpt2cuda::makeVecf(T,4*C);
    l_fch_bias_     = gpt2cuda::makeVecf(4*C);

    l_ln2_means_    = gpt2cuda::makeVecf(B,T);
    l_ln2_rstds_    = gpt2cuda::makeVecf(B,T);
    l_ln2_gamma_    = gpt2cuda::makeVecf(C);
    l_ln2_beta_     = gpt2cuda::makeVecf(C);


    l_fcproj_        = gpt2cuda::makeVecf(B,T,C);
    l_fcproj_weight_ = gpt2cuda::makeVecf(4*C,C);
    l_fcproj_bias_  = gpt2cuda::makeVecf(C);

    l_residual3_ = gpt2cuda::makeVecf(B,T,C);



	dl_residual2 =gpt2cuda::makeZero(B, T, C);
	dl_fcproj =gpt2cuda::makeZero(B, T, C);
	dl_fch_gelu =gpt2cuda::makeZero(B, T, 4 * C);
	dl_fch =gpt2cuda::makeZero(B, T, 4 * C);
	dl_ln2 =gpt2cuda::makeZero(B, T, C);
	dresidual =gpt2cuda::makeZero(B, T, C);
	dl_attproj =gpt2cuda::makeZero(B, T, C);
	dl_atty =gpt2cuda::makeZero(B, T, C);
	dl_qkv =gpt2cuda::makeZero(B, T, 3 * C);
	dl_ln1 =gpt2cuda::makeZero(B, T, C);
    
}

StdVecf Layer::Forward(const StdVecf &residual){

    BatchLayerNormForward(l_ln1_.data(), l_ln1_means_.data(), l_ln1_rstds_.data(), residual.data(), l_ln1_gamma_.data(),l_ln1_beta_.data(),B_,T_,C_);

    BatchMatmulForward(l_qkv_.data(), l_ln1_.data(), qkv_weight_.data(), qkv_bias_.data(), B_,T_,C_,3*C_);

    BatchAttentionForward(l_atty_.data(),l_qkv_.data(),B_,T_,3 * C_);

    BatchMatmulForward(l_attproj_.data(),l_atty_.data(),l_attproj_weight_.data(),l_attproj_bias_.data(),B_,T_,C_,C_); 

    BatchResidualForward(l_residual2_.data(),residual.data(),l_attproj_.data(),B_,T_,C_);

    BatchLayerNormForward(l_ln2_.data(), l_ln2_means_.data(), l_ln2_rstds_.data(), l_residual2_.data(), l_ln2_gamma_.data(), l_ln2_beta_.data(), B_,T_,C_);
    
    BatchMatmulGeluForward(l_fch_gelu_.data(),l_ln2_.data(),l_fch_weight_.data(),l_fch_bias_.data(),B_,T_,C_,4*C_);

    BatchMatmulForward(l_fcproj_.data(),l_fch_gelu_.data(),l_fcproj_weight_.data(),l_fcproj_bias_.data(),B_,T_,4*C_,C_);

    BatchResidualForward(l_residual3_.data(),l_residual2_.data(),l_fcproj_.data(),B_,T_,C_);

    return l_residual3_;
}

StdVecf gpt2cuda::Layer::Backward(const StdVecf&d_outputs, const StdVecf &inputs){
    return d_outputs;
}

