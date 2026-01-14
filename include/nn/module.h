#pragma once
#include "core/tensor.h"
#include <vector>

namespace tf {

struct Param {
  Tensor *value;
  Tensor *grad;
};

struct NamedParam {
  std::string name;
  Tensor *value;
  Tensor *grad;
};

class Module {
public:
  virtual ~Module() = default;
  virtual Tensor forward(const Tensor &x) = 0;
  virtual Tensor backward(const Tensor &grad_out) = 0;

  virtual std::vector<NamedParam> named_parameters() const { return {}; }

  virtual std::vector<Param> params() const {
    std::vector<Param> ps;
    for (auto &np : named_parameters()) {
      ps.push_back({np.value, np.grad});
    }
    return ps;
  }
};

}  