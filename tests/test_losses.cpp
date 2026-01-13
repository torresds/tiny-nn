#include "utils/test_utils.h"
#include "nn/losses.h"
#include "core/tensor.h"
#include <cmath>
#include <iostream>

using namespace tf;

void test_bce_stability() {
    Tensor logits(2, 1);
    logits(0, 0) = 50.0f;  
    logits(1, 0) = -50.0f; 
    
    Tensor targets(2, 1);
    targets(0, 0) = 0.0f;
    targets(1, 0) = 1.0f;
    
    Tensor d_logits;
    float loss = bce_with_logits(logits, targets, d_logits);
    
    
    ASSERT_TRUE(!std::isnan(loss));
    ASSERT_TRUE(!std::isinf(loss));
    
    ASSERT_NEAR(loss, 50.0f, 1.0f);
    
    ASSERT_TRUE(!std::isnan(d_logits(0, 0)));
    ASSERT_TRUE(!std::isnan(d_logits(1, 0)));
}

void test_bce_normal() {
    Tensor logits(1, 1);
    logits(0, 0) = 0.0f; // sigmoid(0) = 0.5
    
    Tensor targets(1, 1);
    targets(0, 0) = 1.0f;
    
    Tensor d_logits;
    float loss = bce_with_logits(logits, targets, d_logits);
    
    // -log(0.5) = 0.6931
    ASSERT_NEAR(loss, 0.693147f, 1e-4f);
}

void test_mse_simple() {
    Tensor preds(2, 2); 
    // [[1, 2], [3, 4]]
    preds(0,0)=1; preds(0,1)=2;
    preds(1,0)=3; preds(1,1)=4;
    
    Tensor targets(2, 2);
    // [[1, 1], [3, 5]]
    targets(0,0)=1; targets(0,1)=1;
    targets(1,0)=3; targets(1,1)=5;
    
    // Diffs: [[0, 1], [0, -1]]
    // Sq:    [[0, 1], [0, 1]]
    // Sum = 2.
    // Mean over batch (2) = 1.0
    
    Tensor d_preds;
    float loss = mse_loss(preds, targets, d_preds);
    
    ASSERT_NEAR(loss, 1.0f, 1e-5f);
    
    // Grads: 2*diff / N
    // N=2
    // d = diff
    // [[0, 1], [0, -1]]
    ASSERT_NEAR(d_preds(0,0), 0.0f, 1e-5f);
    ASSERT_NEAR(d_preds(0,1), 1.0f, 1e-5f);
    ASSERT_NEAR(d_preds(1,0), 0.0f, 1e-5f);
    ASSERT_NEAR(d_preds(1,1), -1.0f, 1e-5f);
}
