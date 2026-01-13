#include "nn/dense.h"
#include "core/math.h"
#include <cassert>
#include "core/error.h"

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
  CHECK(x.cols == W.rows, "Dense forward mismatch: input " << x.shape_str() << " expected cols=" << W.rows);
  x_cache = x;
  Tensor y = matmul(x, W);
  y = add_bias_rowwise(y, b);
  return y;
}

Tensor Dense::backward(const Tensor& grad_out) {
  CHECK(grad_out.cols == W.cols, "Dense backward mismatch: grad_out " << grad_out.shape_str() << " expected cols=" << W.cols);
  CHECK(grad_out.rows == x_cache.rows, "Dense backward mismatch: grad_out " << grad_out.shape_str() << " expected rows=" << x_cache.rows);

  Tensor Xt = transpose(x_cache);
  Tensor dW_cur = matmul(Xt, grad_out);
  dW = add(dW, dW_cur); // accum

  Tensor db_cur = sum_rows(grad_out);
  db = add(db, db_cur); // accum

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
