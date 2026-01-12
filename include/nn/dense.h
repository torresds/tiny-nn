#pragma once
#include "nn/module.h"
#include "core/rng.h"

namespace tf {

class Dense final : public Module {
public:
  Dense(int in_features, int out_features, RNG& rng, bool he_init = true);

  Tensor forward(const Tensor& x) override;
  Tensor backward(const Tensor& grad_out) override;
  std::vector<Param> params() override;

private:
  Tensor W;
  Tensor b;

  Tensor dW;
  Tensor db;

  Tensor x_cache;
};

}
