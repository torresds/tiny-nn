#include "data/dataloader.h"
#include <algorithm>
#include <iostream>

namespace tf {

DataLoader::DataLoader(Dataset &dataset, size_t batch_size, bool shuffle,
                       uint64_t seed)
    : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle),
      rng_((unsigned int)seed), current_idx_(0) {
  indices_.resize(dataset_.size());
  std::iota(indices_.begin(), indices_.end(), 0);
  reset();
}

size_t DataLoader::len() const {
  return (dataset_.size() + batch_size_ - 1) / batch_size_;
}

size_t DataLoader::size() const { return dataset_.size(); }

void DataLoader::reset() {
  current_idx_ = 0;
  if (shuffle_) {
    // Fisher-Yates
    for (size_t i = indices_.size() - 1; i > 0; --i) {
      size_t j = rng_.next_u32() % (i + 1);
      std::swap(indices_[i], indices_[j]);
    }
  }
}

bool DataLoader::next(Tensor &batch_x, Tensor &batch_y) {
  if (current_idx_ >= indices_.size())
    return false;

  size_t end_idx = std::min(current_idx_ + batch_size_, indices_.size());
  size_t actual_batch_size = end_idx - current_idx_;

  if (actual_batch_size == 0)
    return false;

  Sample first = dataset_.get(indices_[current_idx_]);

  batch_x = Tensor((int)actual_batch_size, first.x.cols);
  batch_y = Tensor((int)actual_batch_size, first.y.cols);

  {
    float *bx_ptr = batch_x.data.data();
    const float *s_x = first.x.data.data();
    for (size_t k = 0; k < first.x.size(); ++k)
      bx_ptr[k] = s_x[k];

    float *by_ptr = batch_y.data.data();
    const float *s_y = first.y.data.data();
    for (size_t k = 0; k < first.y.size(); ++k)
      by_ptr[k] = s_y[k];
  }

  for (size_t k = 1; k < actual_batch_size; ++k) {
    size_t data_idx = indices_[current_idx_ + k];
    Sample s = dataset_.get(data_idx);

    float *bx_ptr = batch_x.data.data() + k * batch_x.cols;
    const float *s_x = s.x.data.data();
    for (int j = 0; j < batch_x.cols; ++j)
      bx_ptr[j] = s_x[j];

    float *by_ptr = batch_y.data.data() + k * batch_y.cols;
    const float *s_y = s.y.data.data();
    for (int j = 0; j < batch_y.cols; ++j)
      by_ptr[j] = s_y[j];
  }

  current_idx_ = end_idx;
  return true;
}

}  
