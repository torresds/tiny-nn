#include "core/tensor.h"
#include <sstream>

namespace tf {

std::string Tensor::shape_str() const {
  std::ostringstream oss;
  oss << rows << "x" << cols;
  return oss.str();
}

}