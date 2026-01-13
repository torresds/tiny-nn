#pragma once
#include "core/tensor.h"
#include "nn/module.h"
#include <unordered_map>
#include <vector>

namespace tf {

class Adam {
public:
  // default betas: 0.9, 0.999. eps: 1e-8
  explicit Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
                float eps = 1e-8f)
      : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), t_(0) {}

  void zero_grad(const std::vector<Param> &ps);
  void step(const std::vector<Param> &ps);

private:
  float lr_;
  float beta1_;
  float beta2_;
  float eps_;

  // time step
  int t_;
  
  std::unordered_map<Tensor *, Tensor> m_; // 1st moment
  std::unordered_map<Tensor *, Tensor> v_; // 2nd moment
};

}  
