#include <iostream>
#include <iomanip>
#include <random>
#include <vector>

#include "core/rng.h"
#include "core/tensor.h"
#include "nn/dense.h"
#include "nn/losses.h"
#include "optim/sgd.h"

using namespace tf;

static float eval_mse(const Tensor& preds, const Tensor& targets) {
  float acc = 0.0f;
  for (size_t i = 0; i < preds.size(); ++i) {
    float d = preds.data[i] - targets.data[i];
    acc += d * d;
  }
  return acc / (float)preds.rows;
}

int main() {
  const int N = 256;
  const float w1_true = 3.0f;
  const float w2_true = -2.0f;
  const float b_true  = 0.5f;

  Tensor X(N, 2);
  Tensor Y(N, 1);

  std::mt19937 gen(123);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  std::normal_distribution<float> noise(0.0f, 0.05f);

  for (int i = 0; i < N; ++i) {
    float x1 = uni(gen);
    float x2 = uni(gen);
    float y  = w1_true * x1 + w2_true * x2 + b_true + noise(gen);

    X(i, 0) = x1;
    X(i, 1) = x2;
    Y(i, 0) = y;
  }

  RNG rng(42);
  Dense lin(2, 1, rng, /*he_init=*/false);

  SGD opt(0.05f);

  std::vector<Param> params = lin.params();

  const int epochs = 2000;
  for (int ep = 1; ep <= epochs; ++ep) {
    opt.zero_grad(params);

    Tensor preds = lin.forward(X);

    Tensor d_preds;
    float loss = mse_loss(preds, Y, d_preds);

    lin.backward(d_preds);
    opt.step(params);

    if (ep == 1 || ep % 200 == 0) {
      float mse = eval_mse(preds, Y);
      std::cout << "epoch " << std::setw(4) << ep
                << " | loss(mse_loss) " << std::fixed << std::setprecision(6) << loss
                << " | mse(metric) " << std::fixed << std::setprecision(6) << mse
                << "\n";
    }
  }

  Tensor* W = params[0].value;
  Tensor* b = params[1].value;

  std::cout << "\nTrue:  w1=" << w1_true << " w2=" << w2_true << " b=" << b_true << "\n";
  std::cout << "Learned:\n";
  std::cout << "  w1=" << (*W)(0,0) << "\n";
  std::cout << "  w2=" << (*W)(1,0) << "\n";
  std::cout << "  b =" << (*b)(0,0) << "\n";

  Tensor preds = lin.forward(X);
  std::cout << "\nSamples (pred vs target):\n";
  for (int i = 0; i < 5; ++i) {
    std::cout << "  x=(" << X(i,0) << "," << X(i,1) << ")"
              << " pred=" << preds(i,0)
              << " target=" << Y(i,0) << "\n";
  }

  return 0;
}
