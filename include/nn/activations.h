#pragma once
#include "nn/module.h"

namespace tf {

class ReLU final : public Module {
public:
  Tensor forward(const Tensor& x) override;
  Tensor backward(const Tensor& grad_out) override;

private:
  Tensor x_cache;
};

class Sigmoid final : public Module {
public:
  Tensor forward(const Tensor& x) override;
  Tensor backward(const Tensor& grad_out) override;

private:
  Tensor y_cache;
};

}
