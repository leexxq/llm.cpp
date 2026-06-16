namespace gpt2cuda {
    void AdamW(float* data, float * grad_data, float* m, float* v, size_t size, 
    float lr,float beta1,float beta2,float eps,float weight_decay,int t);
}