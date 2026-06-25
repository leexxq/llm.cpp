#pragma once
namespace gpt2cuda {
    void BatchGeluForward(float * outputs,float const * inputs, int B,int T, int C);
    void BatchGeluBackward(float * d_inputs,float const * d_outputs,float const* inputs,int B,int T,int C);
}