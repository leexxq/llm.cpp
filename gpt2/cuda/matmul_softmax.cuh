#pragma once
#include "devvector.cuh"
namespace gpt2cuda{
    // N is row-major ,T is column-major, Return is row-major , C is considered intermediate dimension of matrix multiplication
    void BatchMatmulNTSoftmaxForward(DevVecf& outputs,DevVecf& workspace, const DevVecf& inputs , const DevVecf& weight, const DevVecf& bias , int B, int T,int C , int  Vp,int V,cudaStream_t stream);
    // N is row-major ,T is column-major, Return is row-major , C is considered intermediate dimension of matrix multiplication
    void BatchMatmulNTSoftmaxForward(float * outputs, float const *  inputs , float const * weight, float const * bias , int B, int T, int C , int  Vp,int V);
    std::size_t GetMatmulSoftmaxWorkSpaceSize(int B ,int T ,int Vp);
}

