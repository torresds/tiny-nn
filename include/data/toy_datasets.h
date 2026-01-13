#pragma once
#include "data/dataset.h"

namespace tf {


TensorDataset make_blobs(int samples, int features, int centers,
                         float cluster_std = 1.0f, int seed = 42);

}  
