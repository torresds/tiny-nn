#include "core/tensor.h"
#include <sstream>

namespace tf {

std::string Tensor::shape_str() const {
  std::ostringstream oss;
  oss << rows << "x" << cols;
  return oss.str();
}

// steal data ptr, leave other empty
Tensor::Tensor(Tensor&& other) noexcept 
  : rows(other.rows), cols(other.cols), data(std::move(other.data)) {
  other.rows = 0;
  other.cols = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    rows = other.rows;
    cols = other.cols;
    data = std::move(other.data);
    // dont leave trash behind
    other.rows = 0;
    other.cols = 0;
  }
  return *this;
}

}