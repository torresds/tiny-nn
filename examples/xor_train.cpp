#include <iostream>
#include <iomanip>
#include "core/rng.h"
#include "nn/dense.h"
#include "nn/activations.h"
#include "nn/losses.h"
#include "optim/sgd.h"

using namespace tf;

static float accuracy_binary(const Tensor& logits, const Tensor& y) {
  int correct = 0;
  for (int i = 0; i < logits.rows; ++i) {
    float pred = (logits(i,0) > 0.0f) ? 1.0f : 0.0f; 
    if ((pred > 0.5f && y(i,0) > 0.5f) || (pred < 0.5f && y(i,0) < 0.5f))
      correct++;
  }
  return (float)correct / (float)logits.rows;
}

int main() {
  RNG rng(42);

  Tensor X(4, 2);
  Tensor Y(4, 1);

  X(0,0)=0; X(0,1)=0; Y(0,0)=0;
  X(1,0)=0; X(1,1)=1; Y(1,0)=1;
  X(2,0)=1; X(2,1)=0; Y(2,0)=1;
  X(3,0)=1; X(3,1)=1; Y(3,0)=0;

  Dense fc1(2, 8, rng, true);
  ReLU  relu1;
  Dense fc2(8, 1, rng, false); 
  SGD opt(0.1f);

  auto p1 = fc1.params();
  auto p2 = fc2.params();
  std::vector<Param> params;
  params.insert(params.end(), p1.begin(), p1.end());
  params.insert(params.end(), p2.begin(), p2.end());

  for (int epoch = 1; epoch <= 5000; ++epoch) {
    opt.zero_grad(params);

    Tensor z1 = fc1.forward(X);
    Tensor a1 = relu1.forward(z1);
    Tensor logits = fc2.forward(a1);

    Tensor dlogits;
    float loss = bce_with_logits(logits, Y, dlogits);

    Tensor da1 = fc2.backward(dlogits);
    Tensor dz1 = relu1.backward(da1);
    Tensor dX  = fc1.backward(dz1);
    (void)dX;
    opt.step(params);

    if (epoch % 250 == 0 || epoch == 1) {
      float acc = accuracy_binary(logits, Y);
      std::cout << "epoch " << std::setw(4) << epoch
                << " | loss " << std::fixed << std::setprecision(6) << loss
                << " | acc " << std::setprecision(3) << acc
                << "\n";
    }
  }

  Tensor z1 = fc1.forward(X);
  Tensor a1 = relu1.forward(z1);
  Tensor logits = fc2.forward(a1);

  std::cout << "\nFinal logits:\n";
  for (int i = 0; i < 4; ++i) {
    std::cout << "  x=(" << X(i,0) << "," << X(i,1) << ") -> logit=" << logits(i,0) << " target=" << Y(i,0) << "\n";
  }

  return 0;
}
