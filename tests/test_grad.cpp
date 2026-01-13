#include "utils/test_utils.h"
#include "nn/dense.h"
#include "nn/losses.h"
#include "core/tensor.h"
#include "core/math.h"
#include <iostream>

using namespace tf;

float compute_loss(const Tensor& y, const Tensor& target) {
    float loss = 0.0f;
    for(size_t i=0; i<y.size(); ++i) {
        float diff = y.data[i] - target.data[i];
        loss += 0.5f * diff * diff;
    }
    return loss;
}

Tensor compute_grad_loss(const Tensor& y, const Tensor& target) {
    return sub(y, target);
}

void test_dense_grad_check() {
    RNG rng(42);
    Tensor x(2, 3);
    x.fill_(0.5f); 

    Tensor target(2, 2);
    target.fill_(1.0f); 

    Dense dense(3, 2, rng);

    Tensor y = dense.forward(x);
    Tensor d_loss = compute_grad_loss(y, target);
    dense.backward(d_loss);
    
    auto params = dense.params();
    
    float epsilon = 1e-3f;
    float tolerance = 1e-4f; 

    std::cout << "Checking gradients for W and b..." << std::endl;

    for (auto& param : params) {
        Tensor* value = param.value;
        Tensor* grad = param.grad;

        for (size_t i = 0; i < value->size(); ++i) {
            float original_val = value->data[i];
            float analytical = grad->data[i];

            value->data[i] = original_val + epsilon;
            Tensor y_plus = dense.forward(x);
            float loss_plus = compute_loss(y_plus, target);

            
            value->data[i] = original_val - epsilon;
            Tensor y_minus = dense.forward(x);
            float loss_minus = compute_loss(y_minus, target);

            value->data[i] = original_val;

            float numerical = (loss_plus - loss_minus) / (2.0f * epsilon);

            
            float num_abs = std::abs(numerical);
            float ana_abs = std::abs(analytical);
            float diff = std::abs(numerical - analytical);
            float rel_error = diff / (num_abs + ana_abs + 1e-10f);

            if (diff > 1e-4f && rel_error > tolerance) {
                 std::cout << "Gradient Check Failed at index " << i 
                           << ". Analytical: " << analytical 
                           << ", Numerical: " << numerical 
                           << ", Diff: " << diff 
                           << ", RelError: " << rel_error << std::endl;
                 ASSERT_TRUE(false);
            }
        }
    }
}
