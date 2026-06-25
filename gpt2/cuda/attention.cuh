#pragma once
namespace gpt2cuda {
    void BatchAttentionForward(float * outputs , float const * inputs,int B,int T ,int C3,int NH);
    void BatchAttentionBackward(float * d_inputs,float const * d_outputs,float const * inputs,int B,int T ,int C3,int NH);
}