#include "io/checkpoint.h"
#include "core/error.h"
#include "nn/module.h"
#include <cstdint>
#include <fstream>
#include <map>
#include <vector>

namespace tf {

static const char MAGIC[] = "TNN1";
static const uint32_t VERSION = 1;

void save_checkpoint(const Module &model, const std::string &path) {
  std::ofstream out(path, std::ios::binary);
  CHECK(out.is_open(), "Could not open file for writing: " << path);

  out.write(MAGIC, 4);
  out.write(reinterpret_cast<const char *>(&VERSION), sizeof(VERSION));

  auto params = const_cast<Module &>(model).named_parameters();
  uint32_t num_tensors = params.size();
  out.write(reinterpret_cast<const char *>(&num_tensors), sizeof(num_tensors));

  for (const auto &p : params) {
    uint32_t name_len = p.name.size();
    out.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
    out.write(p.name.c_str(), name_len);

    int32_t rows = p.value->rows;
    int32_t cols = p.value->cols;
    out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

    out.write(reinterpret_cast<const char *>(p.value->data.data()),
              rows * cols * sizeof(float));
  }
}

void load_checkpoint(Module &model, const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  CHECK(in.is_open(), "Could not open file for reading: " << path);

  char magic[5] = {0};
  in.read(magic, 4);
  CHECK(std::string(magic) == MAGIC,
        "Invalid checkpoint file: wrong magic header");
  CHECK(in.good(), "Failed to read magic header");

  uint32_t version;
  in.read(reinterpret_cast<char *>(&version), sizeof(version));
  CHECK(in.good(), "Failed to read version");
  CHECK(version == VERSION, "Unsupported checkpoint version: " << version);

  auto params = model.named_parameters();
  std::map<std::string, Tensor *> param_map;
  for (auto &p : params) {
    param_map[p.name] = p.value;
  }

  uint32_t num_tensors;
  in.read(reinterpret_cast<char *>(&num_tensors), sizeof(num_tensors));
  CHECK(in.good(), "Failed to read number of tensors");

  for (uint32_t i = 0; i < num_tensors; ++i) {
    uint32_t name_len;
    in.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
    CHECK(in.good(), "Failed to read name length for tensor " << i);
    std::string name(name_len, '\0');
    in.read(&name[0], name_len);
    CHECK(in.good(), "Failed to read name for tensor " << i);

    int32_t rows, cols;
    in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    CHECK(in.good(), "Failed to read rows for tensor " << name);
    in.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    CHECK(in.good(), "Failed to read cols for tensor " << name);

    auto it = param_map.find(name);
    CHECK(it != param_map.end(), "Checkpoint contains parameter '"
                                     << name << "' which is not in the model");

    Tensor *tensor = it->second;
    CHECK(tensor->rows == rows && tensor->cols == cols,
          "Shape mismatch for parameter '"
              << name << "': checkpoint=(" << rows << "," << cols
              << "), model=(" << tensor->rows << "," << tensor->cols << ")");

    in.read(reinterpret_cast<char *>(tensor->data.data()),
            rows * cols * sizeof(float));
    CHECK(in.good(), "Failed to read data for parameter " << name);
  }
}

}  
