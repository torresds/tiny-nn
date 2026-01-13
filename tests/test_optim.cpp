#include "core/tensor.h"
#include "optim/adam.h"
#include "utils/test_utils.h"

using namespace tf;

void test_adam_simple() {
  // Minimize y = (x - 5)^2
  // dy/dx = 2*(x - 5)

  Tensor x(1, 1);
  x.fill_(0.0f); // Start at 0

  Tensor dx(1, 1);
  dx.fill_(0.0f);

  // Param wrapper
  std::vector<Param> params = {Param{&x, &dx}};

  // Adam with lr=0.1
  Adam optim(0.1f);

  for (int i = 0; i < 150; ++i) { // Increased iterations slightly
    optim.zero_grad(params);

    // Forward
    float val = x(0, 0);

    // Backward
    dx(0, 0) = 2.0f * (val - 5.0f);

    optim.step(params);
  }

  // Should be close to 5.0
  ASSERT_NEAR(x(0, 0), 5.0f, 0.05f);
}
