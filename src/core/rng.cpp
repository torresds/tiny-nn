#include "core/rng.h"
#include <cmath>

namespace tf {



void xavier_uniform_(Tensor& W, RNG& rng) {
  const float fan_in  = (float)W.rows;
  const float fan_out = (float)W.cols;
  const float limit = std::sqrt(6.0f / (fan_in + fan_out));
  for (auto& v : W.data) v = rng.uniform(-limit, limit);
}

void he_uniform_(Tensor& W, RNG& rng) {
  const float fan_in = (float)W.rows;
  const float limit = std::sqrt(6.0f / fan_in);
  for (auto& v : W.data) v = rng.uniform(-limit, limit);
}

}
