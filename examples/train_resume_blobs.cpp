#include "core/rng.h"
#include "core/tensor.h"
#include "data/dataloader.h"
#include "data/toy_datasets.h"
#include "nn/activations.h"
#include "nn/dense.h"
#include "nn/losses.h"
#include "nn/sequential.h"
#include "optim/adam.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace tf;

static Tensor slice_rows(const Tensor& src, int row_begin, int row_end) {
  if (row_begin < 0) row_begin = 0;
  if (row_end > src.rows) row_end = src.rows;
  if (row_end < row_begin) row_end = row_begin;

  int out_rows = row_end - row_begin;
  Tensor out(out_rows, src.cols, 0.0f);

  for (int r = 0; r < out_rows; ++r) {
    const float* in_ptr  = src.data.data() + (size_t)(row_begin + r) * (size_t)src.cols;
    float* out_ptr = out.data.data() + (size_t)r * (size_t)out.cols;
    for (int c = 0; c < src.cols; ++c) out_ptr[c] = in_ptr[c];
  }
  return out;
}

static int argmax_row(const Tensor& logits, int r) {
  int best = 0;
  float bestv = logits(r, 0);
  for (int c = 1; c < logits.cols; ++c) {
    float v = logits(r, c);
    if (v > bestv) {
      bestv = v;
      best = c;
    }
  }
  return best;
}

static float accuracy_top1(const Tensor& logits, const Tensor& y_onehot) {
  int correct = 0;
  for (int r = 0; r < logits.rows; ++r) {
    int pred = argmax_row(logits, r);

    int target = 0;
    float best = y_onehot(r, 0);
    for (int c = 1; c < y_onehot.cols; ++c) {
      float v = y_onehot(r, c);
      if (v > best) {
        best = v;
        target = c;
      }
    }

    if (pred == target) correct++;
  }
  return (float)correct / (float)logits.rows;
}

static float eval_loss_acc(Sequential& model, const Tensor& X, const Tensor& Y, float& out_acc) {
  Tensor logits = model.forward(X);
  out_acc = accuracy_top1(logits, Y);

  Tensor d_logits;
  float loss = softmax_cross_entropy_with_logits(logits, Y, d_logits);
  return loss;
}

static void train_one_epoch(
  Sequential& model,
  Adam& optim,
  DataLoader& loader,
  const std::vector<Param>& ps,
  float& out_mean_loss
) {
  float loss_acc = 0.0f;
  int batches = 0;

  Tensor Xb, Yb;
  while (loader.next(Xb, Yb)) {
    optim.zero_grad(ps);

    Tensor logits = model.forward(Xb);

    Tensor d_logits;
    float loss = softmax_cross_entropy_with_logits(logits, Yb, d_logits);
    loss_acc += loss;
    batches++;

    model.backward(d_logits);
    optim.step(ps);
  }

  loader.reset();
  out_mean_loss = (batches > 0) ? (loss_acc / (float)batches) : 0.0f;
}

static Sequential make_model(int in_dim, int hidden, int out_dim, unsigned int seed) {
  RNG rng(seed);
  Sequential model;
  model.add(new Dense(in_dim, hidden, rng));
  model.add(new ReLU());
  model.add(new Dense(hidden, out_dim, rng));
  return model; // moves
}

static float prediction_checksum(Sequential& model, const Tensor& X) {
  Tensor logits = model.forward(X);
  double acc = 0.0;
  for (size_t i = 0; i < logits.size(); ++i) {
    acc += (double)logits.data[i] * (double)(i + 1);
  }
  return (float)acc;
}

int main(int argc, char** argv) {
  //   argv[1] = checkpoint path (default: "blobs_resume.tnn")
  //   argv[2] = epochs_before_save (default: 15)
  //   argv[3] = epochs_after_load (default: 15)
  std::string ckpt_path = (argc >= 2) ? argv[1] : "blobs_resume.tnn";
  int epochs_before = (argc >= 3) ? std::atoi(argv[2]) : 15;
  int epochs_after  = (argc >= 4) ? std::atoi(argv[3]) : 15;

  const int samples = 2000;
  const int features = 2;
  const int classes = 3;
  const float cluster_std = 1.25f;
  const int ds_seed = 42;

  std::cout << "--- Train + Eval + Save + Load + Resume (blobs) ---\n";
  std::cout << "ckpt='" << ckpt_path << "'"
            << " | before=" << epochs_before
            << " | after=" << epochs_after << "\n";

  auto full = make_blobs(samples, features, classes, cluster_std, ds_seed);
  const Tensor& X_all = full.features();
  const Tensor& Y_all = full.targets();

  int n_train = (int)(0.8f * samples);
  int n_val = samples - n_train;

  Tensor X_train = slice_rows(X_all, 0, n_train);
  Tensor Y_train = slice_rows(Y_all, 0, n_train);
  Tensor X_val   = slice_rows(X_all, n_train, n_train + n_val);
  Tensor Y_val   = slice_rows(Y_all, n_train, n_train + n_val);

  TensorDataset train_ds(std::move(X_train), std::move(Y_train));
  TensorDataset val_ds(std::move(X_val), std::move(Y_val));

  const int batch_size = 64;

  Sequential model = make_model(features, /*hidden=*/32, classes, /*seed=*/123);
  Adam optim(0.01f);
  auto ps = model.params();

  DataLoader train_loader(train_ds, batch_size, /*shuffle=*/true, /*seed=*/1337);

  std::cout << "\n[Stage 1] Training for " << epochs_before << " epochs...\n";
  for (int ep = 1; ep <= epochs_before; ++ep) {
    float mean_loss = 0.0f;
    train_one_epoch(model, optim, train_loader, ps, mean_loss);

    float val_acc = 0.0f;
    float val_loss = eval_loss_acc(model, val_ds.features(), val_ds.targets(), val_acc);

    std::cout << "epoch " << ep
              << " | train_loss " << mean_loss
              << " | val_loss " << val_loss
              << " | val_acc " << val_acc
              << "\n";

    if (ep % 5 == 0 || ep == epochs_before) {
      model.save(ckpt_path);
      std::cout << "  [saved] " << ckpt_path << "\n";
    }
  }

  float checksum_before = prediction_checksum(model, val_ds.features());
  std::cout << "\n[Stage 1] checksum(val_logits) = " << checksum_before << "\n";

  std::cout << "\n[Stage 2] Creating fresh model (different init), loading checkpoint...\n";
  Sequential loaded = make_model(features, /*hidden=*/32, classes, /*seed=*/999);
  loaded.load(ckpt_path);

  float checksum_loaded = prediction_checksum(loaded, val_ds.features());
  float diff = std::fabs(checksum_before - checksum_loaded);

  std::cout << "[Stage 2] checksum(val_logits) = " << checksum_loaded
            << " | abs_diff = " << diff << "\n";

  if (diff > 1e-3f) {
    std::cout << "[WARN] checksum mismatch after load. Something is off.\n";
  } else {
    std::cout << "[OK] loaded model matches saved weights.\n";
  }

  Adam optim2(0.01f);
  auto ps2 = loaded.params();
  DataLoader train_loader2(train_ds, batch_size, /*shuffle=*/true, /*seed=*/1337);

  std::cout << "\n[Stage 2] Resuming training for " << epochs_after << " epochs...\n";
  for (int ep = 1; ep <= epochs_after; ++ep) {
    float mean_loss = 0.0f;
    train_one_epoch(loaded, optim2, train_loader2, ps2, mean_loss);

    float val_acc = 0.0f;
    float val_loss = eval_loss_acc(loaded, val_ds.features(), val_ds.targets(), val_acc);

    std::cout << "resume_epoch " << ep
              << " | train_loss " << mean_loss
              << " | val_loss " << val_loss
              << " | val_acc " << val_acc
              << "\n";

    if (ep % 5 == 0 || ep == epochs_after) {
      loaded.save(ckpt_path);
      std::cout << "  [saved] " << ckpt_path << "\n";
    }
  }

  std::cout << "\nDone.\n";
  return 0;
}
