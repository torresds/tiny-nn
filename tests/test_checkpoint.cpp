#include "core/rng.h"
#include "core/tensor.h"
#include "io/checkpoint.h"
#include "nn/dense.h"
#include "nn/sequential.h"
#include "utils/test_utils.h"
#include <cstdio>

using namespace tf;
using namespace tf::test;

void test_save_load() {
  RNG rng(42);
  Sequential model1;
  model1.add(new Dense(10, 5, rng));
  model1.add(new Dense(5, 1, rng));

  std::string checkpoint_path = "test_checkpoint.tnn";

  model1.save(checkpoint_path);

  Sequential model2;
  RNG rng2(123);
  model2.add(new Dense(10, 5, rng2));
  model2.add(new Dense(5, 1, rng2));

  auto params1 = model1.named_parameters();
  auto params2 = model2.named_parameters();

  ASSERT_EQ(params1.size(), params2.size());

  bool different = false;
  for (size_t i = 0; i < params1.size(); ++i) {
    if (std::abs(params1[i].value->data[0] - params2[i].value->data[0]) >
        1e-5) {
      different = true;
      break;
    }
  }
  ASSERT_TRUE(different);

  model2.load(checkpoint_path);

  for (size_t i = 0; i < params1.size(); ++i) {
    NamedParam &p1 = params1[i];
    NamedParam &p2 = params2[i];

    ASSERT_EQ(p1.name, p2.name);
    ASSERT_EQ(p1.value->rows, p2.value->rows);
    ASSERT_EQ(p1.value->cols, p2.value->cols);

    for (size_t k = 0; k < p1.value->data.size(); ++k) {
      ASSERT_NEAR(p1.value->data[k], p2.value->data[k], 1e-7);
    }
  }

  remove(checkpoint_path.c_str());
}
