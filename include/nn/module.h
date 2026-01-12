#pragma once
#include <vector>
#include "core/tensor.h"

namespace tf {

struct Param {
  Tensor* value;
  Tensor* grad;
};

class Module {
public:
  virtual ~Module() = default;
  virtual Tensor forward(const Tensor& x) = 0;
  virtual Tensor backward(const Tensor& grad_out) = 0;
  virtual std::vector<Param> params() { return {}; }
};

}