#include "optim/sgd.h"

namespace tf {

void SGD::zero_grad(const std::vector<Param>& ps) {
  for (auto& p : ps) {
    p.grad->fill_(0.0f);
  }
}

void SGD::step(const std::vector<Param>& ps) {
  for (auto& p : ps) {
    for (size_t i = 0; i < p.value->size(); ++i) {
      p.value->data[i] -= lr_ * p.grad->data[i];
    }
  }
}

}
