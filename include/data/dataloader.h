#pragma once
#include "core/rng.h"
#include "data/dataset.h"
#include <cstdint>
#include <numeric>
#include <vector>

namespace tf {

class DataLoader {
public:
  DataLoader(Dataset &dataset, size_t batch_size, bool shuffle,
             uint64_t seed = 42);

  void reset();

  
  
  bool next(Tensor &batch_x, Tensor &batch_y);

  size_t len() const;  
  size_t size() const; 

private:
  Dataset &dataset_;
  size_t batch_size_;
  bool shuffle_;
  RNG rng_;

  std::vector<size_t> indices_;
  size_t current_idx_;
};

}  
