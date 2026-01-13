#pragma once
#include <vector>
#include <string>
#include <cassert>
#include <cstddef>
#include "core/error.h"

namespace tf {

struct Tensor {
  int rows = 0;
  int cols = 0;
  std::vector<float> data;

  Tensor() = default;
  Tensor(int r, int c, float fill = 0.0f) : rows(r), cols(c), data((size_t)r * (size_t)c, fill) {}

  inline float& operator()(int r, int c) {
    CHECK(r >= 0 && r < rows && c >= 0 && c < cols, 
          "Index out of bounds: (" << r << ", " << c << ") for shape " << shape_str());
    return data[(size_t)r * (size_t)cols + (size_t)c];
  }

  inline const float& operator()(int r, int c) const {
    CHECK(r >= 0 && r < rows && c >= 0 && c < cols, 
          "Index out of bounds: (" << r << ", " << c << ") for shape " << shape_str());
    return data[(size_t)r * (size_t)cols + (size_t)c];
  }

  inline size_t size() const { return data.size(); }

  static Tensor zeros(int r, int c) { return Tensor(r, c, 0.0f); }
  static Tensor ones(int r, int c)  { return Tensor(r, c, 1.0f); }

  void fill_(float v) {
    for (auto& x : data) x = v;
  }

  std::string shape_str() const;
};

}
