#pragma once
#include <vector>
#include "nn/module.h"

namespace tf {

class SGD {
public:
  explicit SGD(float lr) : lr_(lr) {}

  void zero_grad(const std::vector<Param>& ps);
  void step(const std::vector<Param>& ps);

private:
  float lr_;
};

}
