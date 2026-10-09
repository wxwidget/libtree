#include "libtree/c_api.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>

#include "libtree/gbdt.h"

namespace {
thread_local std::string last_error;
libtree::Gbdt& Model(LtModel model) {
  if (model == nullptr) throw std::invalid_argument("null model");
  return *static_cast<libtree::Gbdt*>(model);
}
template <typename Function>
int Guard(Function function) {
  try {
    function();
    last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  // C++ exceptions are the only supported failure boundary; arbitrary
  // non-standard exceptions cannot be produced through the C ABI.
  // GCOVR_EXCL_START
  } catch (...) {
    last_error = "unknown native exception";
    return -1;
  }
  // GCOVR_EXCL_STOP
}
}  // namespace
extern "C" {
const char* LtLastError() { return last_error.c_str(); }
LtModel LtCreate(int trees, int depth, int bins, int min_leaf, double rate,
                 double l2, double min_gain, int binary) {
  LtModel model = nullptr;
  Guard([&] {
    if (binary != 0 && binary != 1)
      throw std::invalid_argument("invalid objective");
    libtree::Parameters p;
    p.num_trees = trees;
    p.max_depth = depth;
    p.max_bins = bins;
    p.min_samples_leaf = min_leaf;
    p.learning_rate = rate;
    p.l2 = l2;
    p.min_gain = min_gain;
    p.objective = binary ? libtree::Objective::kBinaryLogistic
                         : libtree::Objective::kSquaredError;
    model = new libtree::Gbdt(p);
  });
  return model;
}
void LtFree(LtModel model) { delete static_cast<libtree::Gbdt*>(model); }
int LtSetNumThreads(LtModel model, int threads) {
  return Guard([&] { Model(model).SetNumThreads(threads); });
}
int LtFit(LtModel model, const float* x, size_t rows, size_t cols,
          const float* y) {
  return Guard([&] {
    if (y == nullptr || rows == 0) throw std::invalid_argument("empty labels");
    Model(model).Fit({x, rows, cols}, std::vector<float>(y, y + rows));
  });
}
int LtPredict(LtModel model, const float* x, size_t rows, size_t cols,
              float* out) {
  return Guard([&] {
    if (out == nullptr && rows != 0) throw std::invalid_argument("null output");
    Model(model).PredictInto({x, rows, cols}, out);
  });
}
size_t LtLossCount(LtModel model) {
  size_t count = 0;
  Guard([&] { count = Model(model).training_loss().size(); });
  return count;
}
int LtLoss(LtModel model, double* out, size_t count) {
  return Guard([&] {
    const auto& loss = Model(model).training_loss();
    if (count != loss.size() || (out == nullptr && count != 0)) {
      throw std::invalid_argument("invalid loss output");
    }
    if (count != 0) std::copy(loss.begin(), loss.end(), out);
  });
}
}  // extern "C"
