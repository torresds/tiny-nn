#pragma once
#include "core/error.h"
#include "core/tensor.h"
#include <vector>

namespace tf {

struct Sample {
  Tensor x;
  Tensor y;
};

class Dataset {
public:
  virtual ~Dataset() = default;
  virtual size_t size() const = 0;
  virtual Sample get(size_t i) const = 0;
};


class TensorDataset : public Dataset {
public:
  TensorDataset(Tensor x, Tensor y) : x_(std::move(x)), y_(std::move(y)) {
    CHECK(x_.rows == y_.rows, "TensorDataset mismatch rows");
  }

  size_t size() const override { return x_.rows; }

  Sample get(size_t i) const override {
    CHECK(i < x_.rows, "Index out of bounds");

    
    Tensor row_x(1, x_.cols);
    const float *x_ptr = x_.data.data() + i * x_.cols;
    for (int j = 0; j < x_.cols; ++j)
      row_x.data[j] = x_ptr[j];

    
    Tensor row_y(1, y_.cols);
    const float *y_ptr = y_.data.data() + i * y_.cols;
    for (int j = 0; j < y_.cols; ++j)
      row_y.data[j] = y_ptr[j];

    return {std::move(row_x), std::move(row_y)};
  }

  
  const Tensor &features() const { return x_; }
  const Tensor &targets() const { return y_; }

private:
  Tensor x_;
  Tensor y_;
};

}  
