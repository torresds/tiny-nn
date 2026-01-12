#include "nn/activations.h"
#include "core/math.h"

namespace tf {

Tensor ReLU::forward(const Tensor& x) {
  x_cache = x;
  return relu(x);
}

Tensor ReLU::backward(const Tensor& grad_out) {
  return relu_backward(x_cache, grad_out);
}

Tensor Sigmoid::forward(const Tensor& x) {
  y_cache = sigmoid(x);
  return y_cache;
}

Tensor Sigmoid::backward(const Tensor& grad_out) {
  return sigmoid_backward_from_output(y_cache, grad_out);
}

}
