#include "nn/losses.h"
#include <cmath>
#include <cassert>
#include "core/error.h"

namespace tf {

static inline float sigmoid_scalar(float x) {
  if (x >= 0.0f) {
    float z = std::exp(-x);
    return 1.0f / (1.0f + z);
  } else {
    float z = std::exp(x);
    return z / (1.0f + z);
  }
}

float bce_with_logits(const Tensor& logits, const Tensor& targets, Tensor& d_logits) {
  CHECK(logits.rows == targets.rows && logits.cols == targets.cols, 
        "BCE mismatch: logits " << logits.shape_str() << ", targets " << targets.shape_str());
  CHECK(logits.cols == 1, "BCE requires 1 col, got " << logits.cols);

  d_logits = Tensor(logits.rows, logits.cols, 0.0f);

  float loss_sum = 0.0f;
  const float n = (float)logits.rows;

  for (int i = 0; i < logits.rows; ++i) {
    const float x = logits(i, 0);
    const float y = targets(i, 0);

    const float m = (x > 0.0f) ? x : 0.0f;
    loss_sum += m - x * y + std::log1p(std::exp(-std::fabs(x)));

    const float s = sigmoid_scalar(x);
    d_logits(i, 0) = (s - y) / n;
  }

  return loss_sum / n;
}

}
