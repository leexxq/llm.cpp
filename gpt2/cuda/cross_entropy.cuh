namespace gpt2cuda {
void CrossEntropySoftmaxBackward(float * d_logits, float const * probs,int const * targets,int B,int T,int C);
// void CrossEntropyForward( float* losses , float const * probs,int const * targets);
}