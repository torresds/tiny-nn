#include "nn/sequential.h"

#include "io/checkpoint.h"

namespace tf {

void Sequential::add(Module *m) { modules_.push_back(m); }

Sequential::~Sequential() {
  for (auto *m : modules_) {
    delete m;
  }
}

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

std::vector<NamedParam> Sequential::named_parameters() const {
  std::vector<NamedParam> out;
  for (size_t i = 0; i < modules_.size(); ++i) {
    auto local = modules_[i]->named_parameters();
    for (auto &p : local) {
      NamedParam np;
      np.name = std::to_string(i) + "." + p.name;
      np.value = p.value;
      np.grad = p.grad;
      out.push_back(np);
    }
  }
  return out;
}

void Sequential::save(const std::string &path) { save_checkpoint(*this, path); }

void Sequential::load(const std::string &path) { load_checkpoint(*this, path); }

}  
