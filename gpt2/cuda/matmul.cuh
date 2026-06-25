#pragma once

namespace gpt2cuda {

    // N is row-major ,T is column-major, Return is row-major , C is considered intermediate dimension of matrix multiplication
    void BatchMatmulNNForward(float * outputs,float const *  inputs , float const * weight, float const * bias , int B, int T,int  C,int Oc);

    // N is row-major ,T is column-major, Return is row-major , C is considered intermediate dimension of matrix multiplication
    void BatchMatmulNTForward(float * outputs,float const *  inputs , float const * weight, float const * bias , int B, int T,int  C,int Oc);

    // N is row-major ,T is column-major, Return is row-major , C is considered intermediate dimension of matrix multiplication
    void BatchMatmulNNBackward(float * d_inputs, float* d_weight, float* d_bias,float const * d_outputs, float const *  inputs , float const * weight,int B, int T,int  C,int Oc);

    // N is row-major ,T is column-major, Return is row-major , C is considered intermediate dimension of matrix multiplication
    void BatchMatmulNTBackward(float * d_inputs, float* d_weight, float* d_bias,float const * d_outputs, float const *  inputs , float const * weight,int B, int T,int  C,int Oc);

    // void BatchMatmulForward(float * outputs,float const *  inputs , float const * weight, float const * bias , int B, int T,int  C,int Oc);
    // void BatchMatmulBackward(float * d_inputs, float* d_weight, float* d_bias,float const * d_outputs, float const *  inputs , float const * weight,int B, int T,int  C,int Oc);

    void BatchMatmulGeluForward(float * outputs,float const *  inputs , float const * weight, float const * bias , int B, int T,int  C,int Oc);
    void BatchMatmulGeluBackward(float * d_inputs, float* d_weight, float* d_bias,float const * d_outputs, float const *  inputs , float const * weight,int B, int T,int  C,int Oc);
}


// VecBTC CudaMatmul(const VecBTC& inputs,const Matf& weight,const Vecf& bias);