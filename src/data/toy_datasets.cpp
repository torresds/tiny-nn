#include "data/toy_datasets.h"
#include "core/rng.h"
#include <cmath>
#include <vector>

namespace tf {

static float randn(RNG &rng) {
  float u1 = rng.uniform01();
  float u2 = rng.uniform01();
  if (u1 < 1e-6f)
    u1 = 1e-6f;
  return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265f * u2);
}

TensorDataset make_blobs(int samples, int features, int centers,
                         float cluster_std, int seed) {
  RNG rng((unsigned int)seed);

  // [-10, 10]
  Tensor center_locs(centers, features);
  for (size_t i = 0; i < center_locs.size(); ++i) {
    center_locs.data[i] = rng.uniform(-10.0f, 10.0f);
  }

  Tensor X(samples, features);
  Tensor Y(samples, centers);
  Y.fill_(0.0f); 

  for (int i = 0; i < samples; ++i) {
    // uniform01 -> [0, 1). * centers -> [0, centers). floor.
    int c = (int)(rng.uniform01() * centers);
    if (c >= centers)
      c = centers - 1;

    Y(i, c) = 1.0f;

    // X = center + noise
    for (int j = 0; j < features; ++j) {
      float center_val = center_locs(c, j);
      float noise = randn(rng) * cluster_std;
      X(i, j) = center_val + noise;
    }
  }

  return TensorDataset(std::move(X), std::move(Y));
}

}  
