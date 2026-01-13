#include "utils/test_utils.h"
#include "nn/dense.h"
#include "core/tensor.h"
#include <iostream>

using namespace tf;

void test_grad_accumulation() {
    RNG rng(42);
    // 2 inputs, 2 outputs
    Dense dense(2, 2, rng);
    
    // First pass
    Tensor x1(1, 2); x1.fill_(1.0f);
    Tensor d1(1, 2); d1.fill_(0.1f);
    
    dense.forward(x1);
    dense.backward(d1);
    
    auto params = dense.params();
    // Copy gradients after first pass
    Tensor dw1 = *params[0].grad;
    Tensor db1 = *params[1].grad;
    
    // Second pass (dont zero grad)
    Tensor x2(1, 2); x2.fill_(1.0f);
    Tensor d2(1, 2); d2.fill_(0.2f); // double the grad
    
    dense.forward(x2);
    dense.backward(d2);
    
    // Check if accumulated
    Tensor dw_total = *params[0].grad;
    Tensor db_total = *params[1].grad;
    
    // Expected: dw_total = dw1 + dw_new
    // check first element
    float val1 = dw1(0,0);
    float val2 = dw_total(0,0);
    
    // Since gradients are linear w.r.t loss, and we used x=1, 
    // the second grad should be approx 2x the first if everything is linear 
    // but here we just check if it INCREASED significantly or is Sum
    
    // Wait, dense backward: dW = Xt * grad_out
    // Pass 1: x=[1,1], grad=[.1,.1]. dW approx 1*.1 = 0.1
    // Pass 2: x=[1,1], grad=[.2,.2]. dW approx 1*.2 = 0.2
    // Total should be 0.3
    
    // If overwrite happened: total would be 0.2
    
    // Let's assert strictly
    ASSERT_NEAR(val2, val1 * 3.0f, 1e-4f); // 0.1 + 0.2 = 0.3 = 3 * 0.1
}
