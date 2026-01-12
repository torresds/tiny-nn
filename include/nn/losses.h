#pragma once
#include "core/tensor.h"

namespace tf {

float bce_with_logits(const Tensor& logits, const Tensor& targets, Tensor& d_logits);

} 
