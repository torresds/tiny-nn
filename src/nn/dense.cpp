#include "nn/dense.h"
#include "core/math.h"
#include <cassert>

namespace tf {

Dense::Dense(int in_features, int out_features, RNG& rng, bool he_init)
  : W(in_features, out_features, 0.0f),
    b(1, out_features, 0.0f),
    dW(in_features, out_features, 0.0f),
    db(1, out_features, 0.0f) {
  if (he_init) he_uniform_(W, rng);
  else xavier_uniform_(W, rng);
}

Tensor Dense::forward(const Tensor& x) {
  assert(x.cols == W.rows);
  x_cache = x;
  Tensor y = matmul(x, W);
  y = add_bias_rowwise(y, b);
  return y;
}

Tensor Dense::backward(const Tensor& grad_out) {
  assert(grad_out.cols == W.cols);
  assert(grad_out.rows == x_cache.rows);

  Tensor Xt = transpose(x_cache);
  dW = matmul(Xt, grad_out);

  db = sum_rows(grad_out);

  Tensor Wt = transpose(W);
  Tensor dX = matmul(grad_out, Wt);

  return dX;
}

std::vector<Param> Dense::params() {
  return {
    Param{ &W, &dW },
    Param{ &b, &db }
  };
}

}
