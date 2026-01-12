#pragma once
#include "core/tensor.h"

namespace tf {

Tensor matmul(const Tensor& A, const Tensor& B);
Tensor transpose(const Tensor& A);
Tensor add(const Tensor& A, const Tensor& B);
Tensor sub(const Tensor& A, const Tensor& B);
Tensor mul(const Tensor& A, const Tensor& B);
Tensor mul_scalar(const Tensor& A, float s);

Tensor add_bias_rowwise(const Tensor& X, const Tensor& b);
Tensor sum_rows(const Tensor& X);

Tensor relu(const Tensor& X);
Tensor relu_backward(const Tensor& X, const Tensor& dY);

Tensor sigmoid(const Tensor& X);
Tensor sigmoid_backward_from_output(const Tensor& sigmoid_out, const Tensor& dY);

float mean(const Tensor& X);

}
