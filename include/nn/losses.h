#pragma once
#include "core/tensor.h"

namespace tf {

float bce_with_logits(const Tensor &logits, const Tensor &targets,
                      Tensor &d_logits);

float mse_loss(const Tensor &preds, const Tensor &targets, Tensor &d_preds);

float softmax_cross_entropy_with_logits(const Tensor &logits,
                                        const Tensor &targets,
                                        Tensor &d_logits);

}  
