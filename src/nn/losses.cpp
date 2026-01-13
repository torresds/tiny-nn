#include "nn/losses.h"
#include "core/error.h"
#include "core/math.h"
#include <cassert>
#include <cmath>

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

float bce_with_logits(const Tensor &logits, const Tensor &targets,
                      Tensor &d_logits) {
  CHECK(logits.rows == targets.rows && logits.cols == targets.cols,
        "BCE mismatch: logits " << logits.shape_str() << ", targets "
                                << targets.shape_str());
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

float mse_loss(const Tensor &preds, const Tensor &targets, Tensor &d_preds) {
  CHECK(preds.rows == targets.rows && preds.cols == targets.cols,
        "MSE mismatch: " << preds.shape_str() << " vs " << targets.shape_str());

  float loss_sum = 0.0f;

  const float batch_size = (float)preds.rows;

  d_preds = Tensor(preds.rows, preds.cols);

  for (size_t i = 0; i < preds.size(); ++i) {
    float diff = preds.data[i] - targets.data[i];
    loss_sum += diff * diff;

    
    
    d_preds.data[i] = 2.0f * diff / batch_size;
  }

  return loss_sum / batch_size;
}

float softmax_cross_entropy_with_logits(const Tensor &logits,
                                        const Tensor &targets,
                                        Tensor &d_logits) {
  CHECK(logits.rows == targets.rows && logits.cols == targets.cols,
        "softmax_ce mismatch: logits " << logits.shape_str() << " targets "
                                       << targets.shape_str());

  
  Tensor max_logits = rowwise_max(logits);
  Tensor shifted = sub_rowwise(logits, max_logits);
  Tensor exps = exp(shifted);
  Tensor Z = rowwise_sum(exps);
  Tensor log_Z = log(Z);

  
  Tensor log_softmax = sub_rowwise(shifted, log_Z);

  
  float total_loss = 0.0f;
  for (size_t i = 0; i < logits.size(); ++i) {
    total_loss -= targets.data[i] * log_softmax.data[i];
  }
  float mean_loss = total_loss / (float)logits.rows;

  
  
  Tensor probs = div_rowwise(exps, Z);

  
  d_logits = Tensor(logits.rows, logits.cols);
  float scale = 1.0f / (float)logits.rows;

  for (size_t i = 0; i < logits.size(); ++i) {
    d_logits.data[i] = (probs.data[i] - targets.data[i]) * scale;
  }

  return mean_loss;
}

}  
