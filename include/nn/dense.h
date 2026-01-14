#pragma once
#include "core/rng.h"
#include "nn/module.h"

namespace tf {

class Dense final : public Module {
public:
  Dense(int in_features, int out_features, RNG &rng, bool he_init = true);

  Tensor forward(const Tensor &x) override;
  Tensor backward(const Tensor &grad_out) override;
  std::vector<NamedParam> named_parameters() const override;

private:
  Tensor W;
  Tensor b;

  Tensor dW;
  Tensor db;

  Tensor x_cache;
};

}  
