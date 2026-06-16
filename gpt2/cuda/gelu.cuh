#pragma once
namespace gpt2cuda {
    void BatchGeluForward();
    void BatchGeluBackward(float * d_inputs,float const * d_outputs,float const* inputs,int B,int T,int C);
}