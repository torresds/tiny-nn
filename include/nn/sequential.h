#pragma once
#include "nn/module.h"
#include <vector>

namespace tf {

class Sequential : public Module {
public:
  Sequential() = default;
  ~Sequential() override;

  void add(Module *m);

  Tensor forward(const Tensor &x) override;
  Tensor backward(const Tensor &grad_out) override;
  std::vector<NamedParam> named_parameters() const override;

  void save(const std::string &path);
  void load(const std::string &path);

private:
  std::vector<Module *> modules_;
};

}  
