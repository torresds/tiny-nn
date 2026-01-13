#include "core/rng.h"
#include "data/dataloader.h"
#include "data/toy_datasets.h"
#include "nn/activations.h"
#include "nn/dense.h"
#include "nn/losses.h"
#include "nn/sequential.h"
#include "optim/adam.h"
#include <iostream>

using namespace tf;

int main() {
  // 1. Data: 1000 samples, 2 features, 3 classes
  int samples = 1000;
  int features = 2;
  int classes = 3;
  float cluster_std = 1.0f;
  std::cout << "Generating " << samples << " blobs..." << std::endl;

  // Dataset holds data in memory
  auto dataset = make_blobs(samples, features, classes, cluster_std, 42);

  // DataLoader manages batching and shuffling
  DataLoader loader(dataset, 32, true, 42);

  // 2. Model: MLP 2 -> 16 -> 3
  RNG rng(42);
  Sequential model;

  // We leak these pointers for simplicity in this example
  model.add(new Dense(features, 16, rng));
  model.add(new ReLU());
  model.add(new Dense(16, classes, rng));

  // 3. Optimizer
  Adam optim(0.01f);
  auto params = model.params();

  std::cout << "Model params: " << params.size() << std::endl;

  // 4. Train Loop
  int epochs = 20;
  for (int epoch = 0; epoch < epochs; ++epoch) {
    float epoch_loss = 0.0f;
    int batches = 0;
    float correct_pred = 0.0f;
    float total_pred = 0.0f;

    Tensor X, Y;
    while (loader.next(X, Y)) {
      // Zero grad
      optim.zero_grad(params);

      // Forward
      Tensor logits = model.forward(X);

      // Loss
      Tensor d_logits;
      float loss = softmax_cross_entropy_with_logits(logits, Y, d_logits);
      epoch_loss += loss;
      batches++;

      // Accuracy (rough check: max logit index vs max target index)
      // Since we don't have argmax yet, let's just track loss

      // Backward
      model.backward(d_logits);

      // Step
      optim.step(params);
    }
    loader.reset();

    std::cout << "Epoch " << epoch << " | Loss: " << epoch_loss / batches
              << std::endl;
  }

  std::cout << "Done!" << std::endl;
  return 0;
}
