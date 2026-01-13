#include "utils/timer.h"
#include "nn/dense.h"
#include "nn/losses.h"
#include "core/rng.h"
#include "core/math.h"
#include <iostream>

using namespace tf;

void bench_mlp_training(int batch_size, int epochs) {
    std::cout << "--- bench mlp training (batch=" << batch_size << ", epochs=" << epochs << ") ---" << std::endl;
    
    
    int input_dim = 784;
    int hidden1 = 512;
    int hidden2 = 256;
    int output_dim = 10;
    
    RNG rng(1337);
    
    Dense fc1(input_dim, hidden1, rng);
    Dense fc2(hidden1, hidden2, rng);
    Dense fc3(hidden2, output_dim, rng);
    
    Tensor x(batch_size, input_dim);
    x.fill_(0.5f);
    
    Tensor y_target(batch_size, output_dim);
    y_target.fill_(0.0f);
    
    for(int i=0; i<batch_size; ++i) y_target(i, i%output_dim) = 1.0f;
    
    {
        bench::Timer t("total training time");
        
        for (int ep = 0; ep < epochs; ++ep) {
            
            Tensor h1 = relu(fc1.forward(x));
            Tensor h2 = relu(fc2.forward(h1));
            Tensor logits = fc3.forward(h2);
            
            
            Tensor d_logits;
            float loss = mse_loss(logits, y_target, d_logits);
            
            
            
            
            auto p1 = fc1.params(); for(auto& p: p1) p.grad->fill_(0.0f);
            auto p2 = fc2.params(); for(auto& p: p2) p.grad->fill_(0.0f);
            auto p3 = fc3.params(); for(auto& p: p3) p.grad->fill_(0.0f);
            
            
            Tensor d_h2 = fc3.backward(d_logits);
            Tensor d_h2_relu = relu_backward(h2, d_h2);
            Tensor d_h1 = fc2.backward(d_h2_relu);
            Tensor d_h1_relu = relu_backward(h1, d_h1);
            fc1.backward(d_h1_relu);
            
            
            float lr = 0.01f;
            
            
        }
    }
}

int main() {
    bench_mlp_training(64, 10); 
    bench_mlp_training(64, 50); 
    return 0;
}
