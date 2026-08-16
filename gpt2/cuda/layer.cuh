#pragma  once
#include <cstddef>
#include "devvector.cuh"
namespace gpt2cuda{
    class Layer {
        // 3C is GPT2's dim before qkv split

    private:




        DevVecf l_fcproj_;      //(B,T,C)




        DevVecf l_fch_;     //(B,T,4C)
        DevVecf l_fch_gelu_;     //(B,T,4C)

        DevVecf l_ln2_;         //(B,T,C)
        DevVecf l_ln2_means_;   //(B,T)
        DevVecf l_ln2_rstds_;   //(B,T)


        DevVecf l_residual2_;   //(B,T,C)

        DevVecf l_attproj_;     //(B,T,C)

        DevVecf l_atty_;        //(B,T,C)
        DevVecf l_logsumexp_;   //(B,NH,T)

        DevVecf l_qkv_;         //(B,T,3C)

        DevVecf l_ln1_;         //(B,T,C)
        DevVecf l_ln1_means_;         //(B,T)
        DevVecf l_ln1_rstds_;         //(B,T)


        void Init(size_t B, size_t T, size_t C, size_t V, size_t NH);
        size_t B_,T_,C_,V_,NH_;
        size_t params_bytes_;


    public:



        DevVecf l_fcproj_weight ; //(C,4C)
        DevVecf l_fcproj_bias ;  //(C)

        DevVecf l_fch_weight ; //(4C,C)
        DevVecf l_fch_bias ;  //(4C)

        DevVecf l_ln2_gamma;   //(C)
        DevVecf l_ln2_beta ;   //(C)

        DevVecf l_attproj_weight;//(C,C)
        DevVecf l_attproj_bias;//(C)

        DevVecf l_qkv_weight;    //(3C,C)
        DevVecf l_qkv_bias;      //(3C)

        DevVecf l_ln1_gamma;         //(C)
        DevVecf l_ln1_beta;          //(C)

        DevVecf dl_residual2;           //(B,T,C)
        DevVecf dl_fcproj;              //(B,T,C)
        DevVecf dl_fcproj_weight ; //(4C,C)
        DevVecf dl_fcproj_bias;  //(C)

        DevVecf dl_fch_gelu;            //(B,T,4C)


        DevVecf dl_fch;                 //(B,T,4C)
        DevVecf dl_fch_weight;         //(4C,C)
        DevVecf dl_fch_bias;            //(4C)

        DevVecf dl_ln2;                 //(B,T,C)
        DevVecf dl_ln2_gamma;           //(C)
        DevVecf dl_ln2_beta;            //(C)




        DevVecf dl_attproj;             //(B,T,C)
        DevVecf dl_attproj_weight;     //(C,C)
        DevVecf dl_attproj_bias;       //(C)

        DevVecf dl_atty;                //(B,T,C)
        DevVecf dl_logsumexp;           //(B,NH,T)

        DevVecf dl_qkv;                 //(B,T,3C)
        DevVecf dl_qkv_weight;    //(3C,C)
        DevVecf dl_qkv_bias;      //(3C)

        DevVecf dl_ln1;                 //(B,T,C)
        DevVecf dl_ln1_gamma;         //(C)
        DevVecf dl_ln1_beta;          //(C)
    public:
        Layer() {}
        Layer(size_t B, size_t T, size_t C, size_t V, size_t NH):B_(B),T_(T),C_(C),V_(V),NH_(NH) { Init(B, T, C, V, NH); }
        std::size_t GetParamsMemorySize() const{return params_bytes_;}

        void Forward(DevVecf& outputs, const DevVecf&inputs,cudaStream_t stream);
        void Backward(DevVecf& dinputs,const DevVecf&d_outputs, const DevVecf &inputs,cudaStream_t stream);
    };
}