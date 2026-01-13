#include "nn/sequential.h"

namespace tf {

void Sequential::add(Module *m) { modules_.push_back(m); }

Tensor Sequential::forward(const Tensor &x) {
  Tensor out = x;
  for (size_t i = 0; i < modules_.size(); ++i) {
    out = modules_[i]->forward(out);
  }
  return out;
}

Tensor Sequential::backward(const Tensor &grad_out) {
  Tensor grad = grad_out;
  for (int i = (int)modules_.size() - 1; i >= 0; --i) {
    grad = modules_[i]->backward(grad);
  }
  return grad;
}

std::vector<Param> Sequential::params() {
  std::vector<Param> all_params;
  for (auto *m : modules_) {
    auto ps = m->params();
    all_params.insert(all_params.end(), ps.begin(), ps.end());
  }
  return all_params;
}

}  
