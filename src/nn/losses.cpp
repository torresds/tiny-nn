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

float mse_loss(const Tensor& preds, const Tensor& targets, Tensor& d_preds) {
  CHECK(preds.rows == targets.rows && preds.cols == targets.cols, 
        "MSE mismatch: " << preds.shape_str() << " vs " << targets.shape_str());

  float loss_sum = 0.0f;
  const float n = (float)(preds.rows * preds.cols); // divide by total elements? Usually just batch size (rows)

  // Standard MSE: 1/N * sum((y_pred - y_true)^2)
  // But for batch, usually 1/BatchSize * sum_batch(...)
  // Let's stick to "Mean over batch" convention used in BCE.
  // If tensor is (Batch, Features), do we sum features then mean batch?
  // BCE above divides by logits.rows (Batch Size), assuming cols=1.
  
  // Im sticking to: Divide by Batch Size (rows). Sum over features.
  // dL/dy = 2 * (y_pred - y_true) / N
  
  const float batch_size = (float)preds.rows;
  
  d_preds = Tensor(preds.rows, preds.cols);

  for(size_t i=0; i<preds.size(); ++i) {
      float diff = preds.data[i] - targets.data[i];
      loss_sum += diff * diff;
      
      // Gradient: d(diff^2)/dx = 2*diff
      // Mean reduction: * 1/N
      d_preds.data[i] = 2.0f * diff / batch_size;
  }
  
  return loss_sum / batch_size;
}

}
