#include "utils/test_utils.h"
#include "nn/dense.h"
#include "nn/losses.h"
#include "core/tensor.h"
#include "core/math.h"
#include <iostream>

using namespace tf;

// Loss simples para teste: Soma dos quadrados: L = 0.5 * sum((y - target)^2)
float compute_loss(const Tensor& y, const Tensor& target) {
    float loss = 0.0f;
    for(size_t i=0; i<y.size(); ++i) {
        float diff = y.data[i] - target.data[i];
        loss += 0.5f * diff * diff;
    }
    return loss;
}

// Derivada da loss em relação a y: dL/dy = y - target
Tensor compute_grad_loss(const Tensor& y, const Tensor& target) {
    // ret = y - target
    return sub(y, target);
}

void test_dense_grad_check() {
    RNG rng(42);
    // Entrada: Batch 2, Features 3
    Tensor x(2, 3);
    x.fill_(0.5f); // Dummy input

    // Target: Batch 2, Features 2
    Tensor target(2, 2);
    target.fill_(1.0f); // Dummy target

    // Layer: 3 -> 2
    Dense dense(3, 2, rng);

    // 1. Analytical Gradient
    // Forward
    Tensor y = dense.forward(x);
    // Backward
    Tensor d_loss = compute_grad_loss(y, target);
    dense.backward(d_loss);
    
    // Pegar gradientes analíticos
    auto params = dense.params();
    // params[0] = W, params[1] = b
    
    float epsilon = 1e-3f;
    float tolerance = 1e-4f; // Relaxado pois float32 tem precisão limitada

    std::cout << "Checking Gradients for W and b..." << std::endl;

    for (auto& param : params) {
        Tensor* value = param.value;
        Tensor* grad = param.grad;

        for (size_t i = 0; i < value->size(); ++i) {
            float original_val = value->data[i];
            float analytical = grad->data[i];

            // Numerical Gradient (Centered Difference)
            // f(x + h)
            value->data[i] = original_val + epsilon;
            Tensor y_plus = dense.forward(x);
            float loss_plus = compute_loss(y_plus, target);

            // f(x - h)
            value->data[i] = original_val - epsilon;
            Tensor y_minus = dense.forward(x);
            float loss_minus = compute_loss(y_minus, target);

            // Restore
            value->data[i] = original_val;

            float numerical = (loss_plus - loss_minus) / (2.0f * epsilon);

            // Rel Error: |num - ana| / (|num| + |ana| + 1e-10)
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
                 ASSERT_TRUE(false); // Fail test
            }
        }
    }
}
