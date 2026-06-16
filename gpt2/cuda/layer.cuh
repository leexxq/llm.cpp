#pragma  once
#include "global.cuh"
#include <cstddef>
namespace gpt2cuda{
    class Layer {
        // 3C is GPT2's dim before qkv split

    private:
        StdVecf l_residual3_;   //(B,T,C)


        StdVecf l_fcproj_;      //(B,T,C)
        StdVecf l_fcproj_weight_ ; //(4C,C)
        StdVecf l_fcproj_bias_ ;  //(C)


        StdVecf l_fch_gelu_;         //(B,T,4C)
        StdVecf l_fch_weight_ ;
        StdVecf l_fch_bias_ ;  

        StdVecf l_ln2_;         //(B,T,C)
        StdVecf l_ln2_means_;   //(B,T)
        StdVecf l_ln2_rstds_;   //(B,T)
        StdVecf l_ln2_gamma_;   //(C)
        StdVecf l_ln2_beta_ ;   //(C)


        StdVecf l_residual2_;   //(B,T,C)

        StdVecf l_attproj_;     //(B,T,C)

        StdVecf l_atty_;        //(B,T,C)
        StdVecf l_attproj_weight_;//(T,C)
        StdVecf l_attproj_bias_;//(C)
        StdVecf l_qkv_;         //(B,T,3C)
        StdVecf qkv_weight_;    //(C,3*C)
        StdVecf qkv_bias_;      //(3*C)

        StdVecf l_ln1_;         //(B,T,C)
        StdVecf l_ln1_means_;         //(B,T)
        StdVecf l_ln1_rstds_;         //(B,T)
        StdVecf l_ln1_gamma_;         //(C)
        StdVecf l_ln1_beta_;          //(C)


        void Init(size_t B, size_t T, size_t C, size_t V, size_t NH);
        size_t B_,T_,C_;

    public:

        StdVecf dl_residual2;           //(B,T,C)
        StdVecf dl_fcproj;              //(B,T,C)
        StdVecf dl_fch_gelu;            //(B,T,4C)
        StdVecf dl_fch;                 //(B,T,4C)
        StdVecf dl_ln2;                 //(B,T,C)
        StdVecf dresidual;              //(B,T,C)
        StdVecf dl_attproj;             //(B,T,C)
        StdVecf dl_atty;                //(B,T,C)
        StdVecf dl_qkv;                 //(B,T,3C)
        StdVecf dl_ln1;                 //(B,T,C)

    public:
        Layer() {}
        Layer(size_t B, size_t T, size_t C, size_t V, size_t NH):B_(B),T_(T),C_(C) { Init(B, T, C, V, NH); }

        StdVecf Forward(const StdVecf&inputs);

        StdVecf Backward(const StdVecf&d_outputs, const StdVecf &inputs);
    };
}