#pragma  once
#include "pinvector.cuh"
#include <cstddef>
namespace gpt2cuda{
    class Layer {
        // 3C is GPT2's dim before qkv split

    private:


        PinVecf l_fcproj_;      //(B,T,C)


        PinVecf l_fch_;     //(B,T,4C)
        PinVecf l_fch_gelu_;     //(B,T,4C)

        PinVecf l_ln2_;         //(B,T,C)
        PinVecf l_ln2_means_;   //(B,T)
        PinVecf l_ln2_rstds_;   //(B,T)


        PinVecf l_residual2_;   //(B,T,C)

        PinVecf l_attproj_;     //(B,T,C)

        PinVecf l_atty_;        //(B,T,C)
        PinVecf l_logsumexp_;   //(B,NH,T)

        PinVecf l_qkv_;         //(B,T,3C)

        PinVecf l_ln1_;         //(B,T,C)
        PinVecf l_ln1_means_;         //(B,T)
        PinVecf l_ln1_rstds_;         //(B,T)


        void Init(size_t B, size_t T, size_t C, size_t V, size_t NH);
        size_t B_,T_,C_,V_,NH_;
        size_t params_bytes_;


    public:


        PinVecf l_fcproj_weight ; //(C,4C)
        PinVecf l_fcproj_bias ;  //(C)

        PinVecf l_fch_weight ; //(4C,C)
        PinVecf l_fch_bias ;  //(4C)

        PinVecf l_ln2_gamma;   //(C)
        PinVecf l_ln2_beta ;   //(C)

        PinVecf l_attproj_weight;//(C,C)
        PinVecf l_attproj_bias;//(C)

        PinVecf l_qkv_weight;    //(3C,C)
        PinVecf l_qkv_bias;      //(3C)

        PinVecf l_ln1_gamma;         //(C)
        PinVecf l_ln1_beta;          //(C)

        PinVecf dl_residual2;           //(B,T,C)
        PinVecf dl_fcproj;              //(B,T,C)
        PinVecf dl_fcproj_weight ; //(4C,C)
        PinVecf dl_fcproj_bias;  //(C)

        PinVecf dl_fch_gelu;            //(B,T,4C)


        PinVecf dl_fch;                 //(B,T,4C)
        PinVecf dl_fch_weight;         //(4C,C)
        PinVecf dl_fch_bias;            //(4C)

        PinVecf dl_ln2;                 //(B,T,C)
        PinVecf dl_ln2_gamma;           //(C)
        PinVecf dl_ln2_beta;            //(C)



        PinVecf dl_attproj;             //(B,T,C)
        PinVecf dl_attproj_weight;     //(C,C)
        PinVecf dl_attproj_bias;       //(C)

        PinVecf dl_atty;                //(B,T,C)

        PinVecf dl_qkv;                 //(B,T,3C)
        PinVecf dl_qkv_weight;    //(3C,C)
        PinVecf dl_qkv_bias;      //(3C)

        PinVecf dl_ln1;                 //(B,T,C)
        PinVecf dl_ln1_gamma;         //(C)
        PinVecf dl_ln1_beta;          //(C)
    public:
        Layer() {}
        Layer(size_t B, size_t T, size_t C, size_t V, size_t NH):B_(B),T_(T),C_(C),V_(V),NH_(NH) { Init(B, T, C, V, NH); }
        std::size_t GetParamsMemorySize() const{return params_bytes_;}

        void Forward(PinVecf& outputs, const PinVecf&inputs);

        void Backward(PinVecf& dinputs,const PinVecf&d_outputs, const PinVecf &inputs);
    };
}