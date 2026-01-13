#include "utils/test_utils.h"
#include "nn/losses.h"
#include "core/tensor.h"
#include <cmath>
#include <iostream>

using namespace tf;

void test_bce_stability() {
    // Teste com logits extremos para garantir que log-sum-exp não exploda
    // e que logits negativos não causem NaN no log(sigmoid)
    
    Tensor logits(2, 1);
    logits(0, 0) = 50.0f;  // Sigmoid ~ 1.0, se target=0 -> Loss alto mas finito
    logits(1, 0) = -50.0f; // Sigmoid ~ 0.0, se target=1 -> Loss alto mas finito
    
    Tensor targets(2, 1);
    targets(0, 0) = 0.0f;
    targets(1, 0) = 1.0f;
    
    Tensor d_logits;
    float loss = bce_with_logits(logits, targets, d_logits);
    
    // std::cout << "Loss Extreme: " << loss << std::endl;
    
    ASSERT_TRUE(!std::isnan(loss));
    ASSERT_TRUE(!std::isinf(loss));
    
    // Loss deve ser aprox 50.0 em ambos os casos.
    // L = - (y * log(sig) + (1-y)*log(1-sig))
    // Se x=50, y=0 -> -log(1 - 1/(1+e^-50)) approx -log(e^-50) = 50
    ASSERT_NEAR(loss, 50.0f, 1.0f);
    
    // Gradiente não deve ser NaN
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
