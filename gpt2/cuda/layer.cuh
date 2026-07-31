#pragma  once
#include "global.cuh"
#include <cstddef>
namespace gpt2cuda{
    class Layer {
        // 3C is GPT2's dim before qkv split

    private:


        StdVecf l_fcproj_;      //(B,T,C)


        StdVecf l_fch_;     //(B,T,4C)
        StdVecf l_fch_gelu_;     //(B,T,4C)

        StdVecf l_ln2_;         //(B,T,C)
        StdVecf l_ln2_means_;   //(B,T)
        StdVecf l_ln2_rstds_;   //(B,T)


        StdVecf l_residual2_;   //(B,T,C)

        StdVecf l_attproj_;     //(B,T,C)

        StdVecf l_atty_;        //(B,T,C)
        StdVecf l_logsumexp_;   //(B,NH,T)

        StdVecf l_qkv_;         //(B,T,3C)

        StdVecf l_ln1_;         //(B,T,C)
        StdVecf l_ln1_means_;         //(B,T)
        StdVecf l_ln1_rstds_;         //(B,T)


        void Init(size_t B, size_t T, size_t C, size_t V, size_t NH);
        size_t B_,T_,C_,V_,NH_;


    public:


        StdVecf l_fcproj_weight ; //(C,4C)
        StdVecf l_fcproj_bias ;  //(C)

        StdVecf l_fch_weight ; //(4C,C)
        StdVecf l_fch_bias ;  //(4C)

        StdVecf l_ln2_gamma;   //(C)
        StdVecf l_ln2_beta ;   //(C)

        StdVecf l_attproj_weight;//(C,C)
        StdVecf l_attproj_bias;//(C)

        StdVecf l_qkv_weight;    //(3C,C)
        StdVecf l_qkv_bias;      //(3C)

        StdVecf l_ln1_gamma;         //(C)
        StdVecf l_ln1_beta;          //(C)

        StdVecf dl_residual2;           //(B,T,C)
        StdVecf dl_fcproj;              //(B,T,C)
        StdVecf dl_fcproj_weight ; //(4C,C)
        StdVecf dl_fcproj_bias;  //(C)

        StdVecf dl_fch_gelu;            //(B,T,4C)


        StdVecf dl_fch;                 //(B,T,4C)
        StdVecf dl_fch_weight;         //(4C,C)
        StdVecf dl_fch_bias;            //(4C)

        StdVecf dl_ln2;                 //(B,T,C)
        StdVecf dl_ln2_gamma;           //(C)
        StdVecf dl_ln2_beta;            //(C)



        StdVecf dl_attproj;             //(B,T,C)
        StdVecf dl_attproj_weight;     //(C,C)
        StdVecf dl_attproj_bias;       //(C)

        StdVecf dl_atty;                //(B,T,C)

        StdVecf dl_qkv;                 //(B,T,3C)
        StdVecf dl_qkv_weight;    //(3C,C)
        StdVecf dl_qkv_bias;      //(3C)

        StdVecf dl_ln1;                 //(B,T,C)
        StdVecf dl_ln1_gamma;         //(C)
        StdVecf dl_ln1_beta;          //(C)



    public:
        Layer() {}
        Layer(size_t B, size_t T, size_t C, size_t V, size_t NH):B_(B),T_(T),C_(C),V_(V),NH_(NH) { Init(B, T, C, V, NH); }

        void Forward(StdVecf& outputs, const StdVecf&inputs);

        void Backward(StdVecf& dinputs,const StdVecf&d_outputs, const StdVecf &inputs);
    };
}