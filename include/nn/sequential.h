#pragma once
#include "nn/module.h"
#include <vector>

namespace tf {

class Sequential : public Module {
public:
  Sequential() = default;
  

  void add(Module *m);

  Tensor forward(const Tensor &x) override;
  Tensor backward(const Tensor &grad_out) override;
  std::vector<Param> params() override;

private:
  std::vector<Module *> modules_;
};

}  
