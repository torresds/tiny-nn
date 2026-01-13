#include "optim/adam.h"
#include <cmath>
#include <iostream>

namespace tf {

void Adam::zero_grad(const std::vector<Param> &ps) {
  for (const auto &p : ps) {
    if (p.grad) {
      p.grad->fill_(0.0f);
    }
  }
}

void Adam::step(const std::vector<Param> &ps) {
  t_++;


  float correction1 = 1.0f - std::pow(beta1_, t_);
  float correction2 = 1.0f - std::pow(beta2_, t_);

  for (const auto &p : ps) {
    if (!p.grad || !p.value)
      continue;

    Tensor *param = p.value;
    Tensor *grad = p.grad;

    if (m_.find(param) == m_.end()) {
      m_[param] = Tensor(param->rows, param->cols, 0.0f);
      v_[param] = Tensor(param->rows, param->cols, 0.0f);
    }

    Tensor &m = m_[param];
    Tensor &v = v_[param];

    // sanity check
    if (grad->rows != param->rows || grad->cols != param->cols)
      continue;

    size_t len = param->size();
    float *p_data = param->data.data();
    const float *g_data = grad->data.data();
    float *m_data = m.data.data();
    float *v_data = v.data.data();

    for (size_t i = 0; i < len; ++i) {
      float g = g_data[i];

      // m = beta1 * m + (1 - beta1) * g
      m_data[i] = beta1_ * m_data[i] + (1.0f - beta1_) * g;

      // v = beta2 * v + (1 - beta2) * g * g
      v_data[i] = beta2_ * v_data[i] + (1.0f - beta2_) * g * g;

      float m_hat = m_data[i] / correction1;
      float v_hat = v_data[i] / correction2;

      p_data[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
    }
  }
}

}  
