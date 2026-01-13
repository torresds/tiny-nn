#include "core/tensor.h"
#include "nn/dense.h"
#include "utils/test_utils.h"
#include <iostream>

using namespace tf;

void test_grad_accumulation() {
  RNG rng(42);
  
  Dense dense(2, 2, rng);

  
  Tensor x1(1, 2);
  x1.fill_(1.0f);
  Tensor d1(1, 2);
  d1.fill_(0.1f);

  dense.forward(x1);
  dense.backward(d1);

  auto params = dense.params();
  
  Tensor dw1 = *params[0].grad;
  Tensor db1 = *params[1].grad;

  
  Tensor x2(1, 2);
  x2.fill_(1.0f);
  Tensor d2(1, 2);
  d2.fill_(0.2f); 

  dense.forward(x2);
  dense.backward(d2);

  
  Tensor dw_total = *params[0].grad;
  Tensor db_total = *params[1].grad;

  
  
  float val1 = dw1(0, 0);
  float val2 = dw_total(0, 0);

  ASSERT_NEAR(val2, val1 * 3.0f, 1e-4f); 
}
