#pragma once
#include "core/tensor.h"

namespace tf {

// Returns mean loss over batch
float bce_with_logits(const Tensor &logits, const Tensor &targets,
                      Tensor &d_logits);

// Returns mean squared error
float mse_loss(const Tensor &preds, const Tensor &targets, Tensor &d_preds);

// Returns mean loss over batch. Logic: LogSumExp for stability.
// targets must be one-hot (N, C)
float softmax_cross_entropy_with_logits(const Tensor &logits,
                                        const Tensor &targets,
                                        Tensor &d_logits);

} // namespace tf
