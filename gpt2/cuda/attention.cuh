#pragma once
namespace gpt2cuda {
    void BatchAttentionForward(float * outputs , float const * inputs,int B,int T ,int C3);
}