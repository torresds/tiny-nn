#include "core/tensor.h"
#include "nn/losses.h"
#include "utils/test_utils.h"
#include <cmath>
#include <iostream>

using namespace tf;

void test_bce_stability() {
  Tensor logits(2, 1);
  logits(0, 0) = 50.0f;
  logits(1, 0) = -50.0f;

  Tensor targets(2, 1);
  targets(0, 0) = 0.0f;
  targets(1, 0) = 1.0f;

  Tensor d_logits;
  float loss = bce_with_logits(logits, targets, d_logits);

  ASSERT_TRUE(!std::isnan(loss));
  ASSERT_TRUE(!std::isinf(loss));

  ASSERT_NEAR(loss, 50.0f, 1.0f);

  ASSERT_TRUE(!std::isnan(d_logits(0, 0)));
  ASSERT_TRUE(!std::isnan(d_logits(1, 0)));
}

void test_bce_normal() {
  Tensor logits(1, 1);
  logits(0, 0) = 0.0f; // sigmoid(0) = 0.5

  Tensor targets(1, 1);
  targets(0, 0) = 1.0f;

  Tensor d_logits;
  float loss = bce_with_logits(logits, targets, d_logits);

  // -log(0.5) = 0.6931
  ASSERT_NEAR(loss, 0.693147f, 1e-4f);
}

void test_mse_simple() {
  Tensor preds(2, 2);
  // [[1, 2], [3, 4]]
  preds(0, 0) = 1;
  preds(0, 1) = 2;
  preds(1, 0) = 3;
  preds(1, 1) = 4;

  Tensor targets(2, 2);
  // [[1, 1], [3, 5]]
  targets(0, 0) = 1;
  targets(0, 1) = 1;
  targets(1, 0) = 3;
  targets(1, 1) = 5;

  // Diffs: [[0, 1], [0, -1]]
  // Sq:    [[0, 1], [0, 1]]
  // Sum = 2.
  // Mean over batch (2) = 1.0

  Tensor d_preds;
  float loss = mse_loss(preds, targets, d_preds);

  ASSERT_NEAR(loss, 1.0f, 1e-5f);

  // Grads: 2*diff / N
  // N=2
  // d = diff
  // [[0, 1], [0, -1]]
  ASSERT_NEAR(d_preds(0, 0), 0.0f, 1e-5f);
  ASSERT_NEAR(d_preds(0, 1), 1.0f, 1e-5f);
  ASSERT_NEAR(d_preds(1, 0), 0.0f, 1e-5f);
  ASSERT_NEAR(d_preds(1, 1), -1.0f, 1e-5f);
}

void test_softmax_ce_sanity() {
  // 2 samples, 3 classes
  Tensor logits(2, 3);
  logits.fill_(0.0f);
  // Sample 0: [0, 0, 0] -> Softmax: [0.33, 0.33, 0.33]
  // Sample 1: [0, 0, 0]

  Tensor targets(2, 3); // One hot
  targets.fill_(0.0f);
  targets(0, 0) = 1.0f; // Target class 0
  targets(1, 2) = 1.0f; // Target class 2

  Tensor d_logits;
  float loss = softmax_cross_entropy_with_logits(logits, targets, d_logits);

  // Expected loss: -log(1/3) = 1.0986
  ASSERT_NEAR(loss, 1.0986f, 1e-4f);

  // Gradient: (p - y)/batch
  // p = 0.333, y=1 (correct class) -> -0.666 / 2 = -0.333
  // p = 0.333, y=0 (wrong class)   ->  0.333 / 2 =  0.166

  ASSERT_NEAR(d_logits(0, 0), -0.3333f, 1e-3f);
  ASSERT_NEAR(d_logits(0, 1), 0.1666f, 1e-3f);
}

void test_softmax_ce_stability() {
  Tensor logits(1, 3);
  // logits: [1000, 0, 0], Target: 0 (index 0)
  // Shifted: [0, -1000, -1000]
  // Exp: [1, 0, 0] -> Sum=1 -> LogSum=0
  // LogSoftmax: [0, -1000, -1000]
  // Loss: -1 * 0 = 0
  logits(0, 0) = 1000.0f;
  logits(0, 1) = 0.0f;
  logits(0, 2) = 0.0f;

  Tensor targets(1, 3, 0.0f);
  targets(0, 0) = 1.0f;

  Tensor d_logits;
  float loss = softmax_cross_entropy_with_logits(logits, targets, d_logits);

  ASSERT_NEAR(loss, 0.0f, 1e-4f);
  ASSERT_TRUE(!std::isnan(loss));
  ASSERT_TRUE(!std::isinf(loss));
}

void test_softmax_ce_grad_check() {
  Tensor logits(1, 3);
  logits(0, 0) = 1.0f;
  logits(0, 1) = -2.0f;
  logits(0, 2) = 0.5f;

  Tensor targets(1, 3, 0.0f);
  targets(0, 2) = 1.0f; // Class 2

  Tensor d_logits;
  softmax_cross_entropy_with_logits(logits, targets, d_logits);

  float h = 1e-3f;
  // Check grad for logits(0,0)
  // f(x+h)
  logits(0, 0) += h;
  Tensor temp_d;
  float loss_plus = softmax_cross_entropy_with_logits(logits, targets, temp_d);

  // f(x-h)
  logits(0, 0) -= 2 * h;
  float loss_minus = softmax_cross_entropy_with_logits(logits, targets, temp_d);

  float num_grad = (loss_plus - loss_minus) / (2 * h);
  float ana_grad = d_logits(0, 0);

  ASSERT_NEAR(num_grad, ana_grad, 1e-3f);
}
