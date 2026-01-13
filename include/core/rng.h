#pragma once
#include "core/tensor.h"

namespace tf {

struct RNG {
  unsigned int state = 12345u;
  explicit RNG(unsigned int seed = 12345u) : state(seed) {}

  inline unsigned int next_u32() {
    state = 1664525u * state + 1013904223u;
    return state;
  }

  inline float uniform01() {
    return (next_u32() >> 8) * (1.0f / 16777216.0f);
  }

  inline float uniform(float a, float b) {
    return a + (b - a) * uniform01();
  }
};

void xavier_uniform_(Tensor& W, RNG& rng);
void he_uniform_(Tensor& W, RNG& rng);

}
