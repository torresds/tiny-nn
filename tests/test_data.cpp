#include "data/dataloader.h"
#include "data/toy_datasets.h"
#include "utils/test_utils.h"

using namespace tf;

void test_make_blobs() {
  int n = 50;
  int features = 2;
  int centers = 3;
  auto ds = make_blobs(n, features, centers);

  ASSERT_EQ(ds.size(), (size_t)n);
  Sample s = ds.get(0);
  ASSERT_EQ(s.x.rows, 1);
  ASSERT_EQ(s.x.cols, features);
  ASSERT_EQ(s.y.rows, 1);
  ASSERT_EQ(s.y.cols, centers);
}

void test_dataloader_batching() {
  // 10 samples, batch 3
  // batches: 3, 3, 3, 1
  int n = 10;
  auto ds = make_blobs(n, 2, 2);

  DataLoader loader(ds, 3, false);

  ASSERT_EQ(loader.len(), (size_t)4);

  Tensor X, Y;

  // Batch 1
  ASSERT_TRUE(loader.next(X, Y));
  ASSERT_EQ(X.rows, 3);

  // Batch 2
  ASSERT_TRUE(loader.next(X, Y));
  ASSERT_EQ(X.rows, 3);

  // Batch 3
  ASSERT_TRUE(loader.next(X, Y));
  ASSERT_EQ(X.rows, 3);

  // Batch 4
  ASSERT_TRUE(loader.next(X, Y));
  ASSERT_EQ(X.rows, 1);

  // End
  ASSERT_TRUE(!loader.next(X, Y));

  // Reset
  loader.reset();
  ASSERT_TRUE(loader.next(X, Y));
  ASSERT_EQ(X.rows, 3);
}
