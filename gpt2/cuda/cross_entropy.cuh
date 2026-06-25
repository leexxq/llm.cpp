namespace gpt2cuda {
    void BatchCrossEntropySoftmaxBackward(float * d_logits, float const * probs,int const * targets,int B,int T,int V,int Vp,float scale);
    void BatchCrossEntropyForward(float* losses, float const * inputs, int const * targets,int B,int T,int Vp);
// void CrossEntropyForward( float* losses , float const * probs,int const * targets);
}