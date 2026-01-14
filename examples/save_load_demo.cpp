#include "core/rng.h"
#include "data/toy_datasets.h"
#include "io/checkpoint.h"
#include "nn/activations.h"
#include "nn/dense.h"
#include "nn/losses.h"
#include "nn/sequential.h"
#include "optim/sgd.h"
#include <iostream>
#include <vector>

using namespace tf;

int main() {
  std::cout << "--- Save/Load Demo ---" << std::endl;

  std::cout << "[Step 1] Training Initial Model..." << std::endl;
  RNG rng(42);
  auto dataset = make_blobs(100, 2, 2, 1.0f, 42); // int seed
  const Tensor &X_train = dataset.features();
  const Tensor &y_train = dataset.targets();

  Sequential model;
  model.add(new Dense(2, 4, rng));
  model.add(new ReLU());
  model.add(new Dense(4, 2, rng));

  SGD optimizer(0.1f);

  auto ps = model.params();
  for (int epoch = 0; epoch < 50; ++epoch) {
    optimizer.zero_grad(ps);

    Tensor y_pred = model.forward(X_train);

    Tensor grad_loss;
    float loss = softmax_cross_entropy_with_logits(y_pred, y_train, grad_loss);

    model.backward(grad_loss);

    optimizer.step(ps);

    if (epoch % 10 == 0) {
      // std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
    }
  }

  Tensor pred_before = model.forward(X_train);

  std::cout << "[Step 2] Saving model to 'demo_model.tnn'..." << std::endl;
  model.save("demo_model.tnn");

  std::cout << "[Step 3] Creating fresh model (random weights)..." << std::endl;
  Sequential loaded_model;
  RNG rng_fresh(999);
  loaded_model.add(new Dense(2, 4, rng_fresh));
  loaded_model.add(new ReLU());
  loaded_model.add(new Dense(4, 2, rng_fresh));

  std::cout << "[Step 4] Loading checkpoint..." << std::endl;
  loaded_model.load("demo_model.tnn");

  std::cout << "[Step 5] Verifying predictions..." << std::endl;
  Tensor pred_after = loaded_model.forward(X_train);

  float diff = 0.0f;
  for (size_t i = 0; i < pred_before.size(); ++i) {
    diff += std::abs(pred_before.data[i] - pred_after.data[i]);
  }

  std::cout << "Difference between original and loaded predictions: " << diff
            << std::endl;
  if (diff < 1e-4) {
    std::cout << "SUCCESS: Predictions match!" << std::endl;
  } else {
    std::cout << "FAILURE: Predictions do not match!" << std::endl;
    return 1;
  }
  return 0;
}
