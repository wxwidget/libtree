#include "libtree/c_api.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "libtree/c5.h"
#include "libtree/gbdt.h"

namespace {
thread_local std::string last_error;
libtree::Gbdt& Model(LtModel model) {
  if (model == nullptr) throw std::invalid_argument("null model");
  return *static_cast<libtree::Gbdt*>(model);
}
libtree::C5Classifier& C5Model(LtC5Model model) {
  if (model == nullptr) throw std::invalid_argument("null C5 model");
  return *static_cast<libtree::C5Classifier*>(model);
}
std::vector<std::string> FeatureNames(const char* const* names,
                                      size_t name_count) {
  if (names == nullptr && name_count != 0)
    throw std::invalid_argument("null feature names");
  std::vector<std::string> labels;
  labels.reserve(name_count);
  for (size_t i = 0; i < name_count; ++i) {
    if (names[i] == nullptr) throw std::invalid_argument("null feature name");
    labels.emplace_back(names[i]);
  }
  return labels;
}
template <typename Function>
void Inspection(const char* const* names, size_t name_count, char* out,
                size_t capacity, size_t* required, Function render) {
  if (required == nullptr) throw std::invalid_argument("null required size");
  const auto labels = FeatureNames(names, name_count);
  std::ostringstream stream;
  render(stream, labels);
  const std::string text = stream.str();
  *required = text.size() + 1;
  if (out == nullptr && capacity == 0) return;
  if (out == nullptr || capacity < *required)
    throw std::invalid_argument("inspection output buffer too small or null");
  std::memcpy(out, text.c_str(), *required);
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
    // The C ABI translates standard exceptions; this fallback protects the
    // boundary if a non-standard exception escapes a native operation.
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
int LtSerialize(LtModel model, char* out, size_t capacity, size_t* required) {
  return Guard([&] {
    Inspection(nullptr, 0, out, capacity, required,
               [&](std::ostream& output, const std::vector<std::string>&) {
                 Model(model).SaveModel(output);
               });
  });
}
LtModel LtDeserialize(const char* data, size_t size, int threads) {
  LtModel result = nullptr;
  Guard([&] {
    if (data == nullptr || size == 0)
      throw std::invalid_argument("empty model snapshot");
    auto model = std::make_unique<libtree::Gbdt>();
    model->SetNumThreads(threads);
    std::istringstream input(std::string(data, size));
    model->LoadModel(input);
    result = model.release();
  });
  return result;
}
int LtDumpModel(LtModel model, int format, int tree_index,
                const char* const* names, size_t name_count, char* out,
                size_t capacity, size_t* required) {
  return Guard([&] {
    Inspection(
        names, name_count, out, capacity, required,
        [&](std::ostream& output, const std::vector<std::string>& labels) {
          Model(model).DumpModel(output,
                                 static_cast<libtree::DumpFormat>(format),
                                 tree_index, labels);
        });
  });
}
int LtExplainPrediction(LtModel model, const float* x, size_t cols, int format,
                        const char* const* names, size_t name_count, char* out,
                        size_t capacity, size_t* required) {
  return Guard([&] {
    Inspection(
        names, name_count, out, capacity, required,
        [&](std::ostream& output, const std::vector<std::string>& labels) {
          Model(model).ExplainPrediction(
              {x, 1, cols}, output, static_cast<libtree::DumpFormat>(format),
              labels);
        });
  });
}
LtC5Model LtC5Create(int rounds, int depth, int bins, int min_leaf, double rate,
                     double alpha, double min_gain, int criterion,
                     int threads) {
  LtC5Model model = nullptr;
  Guard([&] {
    libtree::C5Parameters p;
    p.num_estimators = rounds;
    p.max_depth = depth;
    p.max_bins = bins;
    p.min_samples_leaf = min_leaf;
    p.learning_rate = rate;
    p.pruning_alpha = alpha;
    p.min_gain = min_gain;
    p.criterion = static_cast<libtree::SplitCriterion>(criterion);
    p.num_threads = threads;
    model = new libtree::C5Classifier(p);
  });
  return model;
}
void LtC5Free(LtC5Model model) {
  delete static_cast<libtree::C5Classifier*>(model);
}
int LtC5Serialize(LtC5Model model, char* out, size_t capacity,
                  size_t* required) {
  return Guard([&] {
    Inspection(nullptr, 0, out, capacity, required,
               [&](std::ostream& output, const std::vector<std::string>&) {
                 C5Model(model).SaveModel(output);
               });
  });
}
LtC5Model LtC5Deserialize(const char* data, size_t size, int threads) {
  LtC5Model result = nullptr;
  Guard([&] {
    if (data == nullptr || size == 0)
      throw std::invalid_argument("empty C5 snapshot");
    libtree::C5Parameters parameters;
    parameters.num_threads = threads;
    auto model = std::make_unique<libtree::C5Classifier>(parameters);
    std::istringstream input(std::string(data, size));
    model->LoadModel(input);
    result = model.release();
  });
  return result;
}
int LtC5Fit(LtC5Model model, const float* x, size_t rows, size_t cols,
            const int* y, const double* weights) {
  return Guard([&] {
    if (y == nullptr || rows == 0 || rows > 100000000)
      throw std::invalid_argument("invalid C5 labels");
    C5Model(model).Fit({x, rows, cols}, std::vector<int>(y, y + rows),
                       weights ? std::vector<double>(weights, weights + rows)
                               : std::vector<double>{});
  });
}
size_t LtC5ClassCount(LtC5Model model) {
  size_t count = 0;
  Guard([&] { count = C5Model(model).num_classes(); });
  return count;
}
int LtC5PredictProba(LtC5Model model, const float* x, size_t rows, size_t cols,
                     double* out) {
  return Guard([&] { C5Model(model).PredictProbaInto({x, rows, cols}, out); });
}
int LtC5Predict(LtC5Model model, const float* x, size_t rows, size_t cols,
                int* out) {
  return Guard([&] { C5Model(model).PredictInto({x, rows, cols}, out); });
}
size_t LtC5ErrorCount(LtC5Model model) {
  size_t count = 0;
  Guard([&] { count = C5Model(model).estimator_errors().size(); });
  return count;
}
int LtC5Errors(LtC5Model model, double* out, size_t count) {
  return Guard([&] {
    const auto& errors = C5Model(model).estimator_errors();
    if (count != errors.size() || (count != 0 && out == nullptr))
      throw std::invalid_argument("invalid C5 error output");
    std::copy(errors.begin(), errors.end(), out);
  });
}
int LtC5DumpModel(LtC5Model model, int format, int tree_index,
                  const char* const* names, size_t name_count, char* out,
                  size_t capacity, size_t* required) {
  return Guard([&] {
    Inspection(
        names, name_count, out, capacity, required,
        [&](std::ostream& output, const std::vector<std::string>& labels) {
          C5Model(model).DumpModel(output,
                                   static_cast<libtree::DumpFormat>(format),
                                   tree_index, labels);
        });
  });
}
int LtC5DumpRules(LtC5Model model, int format, const char* const* names,
                  size_t name_count, char* out, size_t capacity,
                  size_t* required) {
  return Guard([&] {
    Inspection(
        names, name_count, out, capacity, required,
        [&](std::ostream& output, const std::vector<std::string>& labels) {
          C5Model(model).DumpRules(
              output, static_cast<libtree::DumpFormat>(format), labels);
        });
  });
}
int LtC5Explain(LtC5Model model, const float* x, size_t cols, int format,
                const char* const* names, size_t name_count, char* out,
                size_t capacity, size_t* required) {
  return Guard([&] {
    Inspection(
        names, name_count, out, capacity, required,
        [&](std::ostream& output, const std::vector<std::string>& labels) {
          C5Model(model).Explain({x, 1, cols}, output,
                                 static_cast<libtree::DumpFormat>(format),
                                 labels);
        });
  });
}
}  // extern "C"
