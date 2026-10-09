#include "libtree/gbdt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
int checks = 0;
void Check(bool condition) {
  ++checks;
  if (!condition) throw std::runtime_error("check " + std::to_string(checks));
}
template <typename Callable>
void Throws(Callable callable) {
  bool threw = false;
  try {
    callable();
  } catch (const std::exception&) {
    threw = true;
  }
  Check(threw);
}
// Independent, ordinary traversal of the portable model format. This checks
// optimized prediction layouts without assuming their in-memory representation.
void CheckSerializedPrediction(const libtree::Gbdt& model,
                               libtree::MatrixView input) {
  std::stringstream stream;
  model.SaveModel(stream);
  std::string magic;
  int version, objective, trees, depth, bins, min_leaf;
  double rate, l2, gain, base;
  std::size_t features, count;
  stream >> magic >> version >> objective >> trees >> depth >> bins >>
      min_leaf >> rate >> l2 >> gain >> features >> base >> count;
  std::vector<double> margins(input.rows, base);
  for (std::size_t t = 0; t < count; ++t) {
    struct ReferenceNode {
      int feature, left, right;
      float threshold;
      double value;
      bool missing_left;
    };
    std::size_t nodes;
    stream >> nodes;
    std::vector<ReferenceNode> tree(nodes);
    for (auto& node : tree)
      stream >> node.feature >> node.left >> node.right >> node.threshold >>
          node.value >> node.missing_left;
    for (std::size_t row = 0; row < input.rows; ++row) {
      int index = 0;
      while (tree[index].feature >= 0) {
        const auto& node = tree[index];
        const float value = input.data[row * input.cols + node.feature];
        index =
            (std::isnan(value) ? node.missing_left : value <= node.threshold)
                ? node.left
                : node.right;
      }
      margins[row] += rate * tree[index].value;
    }
  }
  auto actual = model.Predict(input);
  std::vector<float> direct(input.rows);
  model.PredictInto(input, direct.data());
  Check(actual == direct);
  for (std::size_t row = 0; row < input.rows; ++row) {
    double expected = margins[row];
    if (objective == 1) {
      const double e = std::exp(-std::abs(expected));
      expected = expected >= 0 ? 1 / (1 + e) : e / (1 + e);
    }
    Check(actual[row] == static_cast<float>(expected));
  }
}
void Run() {
  using libtree::Gbdt;
  using libtree::Parameters;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  Parameters p;
  p.num_trees = 50;
  p.min_samples_leaf = 1;
  p.learning_rate = 0.2;
  p.l2 = 0;
  std::vector<float> x{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> y{0, 0, 0, 0, 10, 10, 10, 10};
  Gbdt model(p);
  Throws([&] { model.Predict({x.data(), 8, 1}); });
  model.Fit({x.data(), 8, 1}, y);
  auto prediction = model.Predict({x.data(), 8, 1});
  Check(prediction.size() == y.size());
  for (std::size_t i = 0; i < y.size(); ++i)
    Check(std::abs(prediction[i] - y[i]) < 0.001);
  Check(model.training_loss().back() < model.training_loss().front());
  Check(model.num_trees() == 50);
  Gbdt second(p);
  second.Fit({x.data(), 8, 1}, y);
  Check(second.Predict({x.data(), 8, 1}) == prediction);
  model.Fit({x.data(), 8, 1}, std::vector<float>(8, 3));
  for (float value : model.Predict({x.data(), 8, 1})) Check(value == 3);
  Check(model.Predict({nullptr, 0, 1}).empty());
  Throws([&] { model.Predict({x.data(), 8, 2}); });
  Throws([&] { model.Fit({nullptr, 0, 1}, {}); });
  Throws([&] { model.Fit({x.data(), 8, 1}, {1}); });
  Throws([&] { model.Fit({nullptr, 8, 1}, y); });
  auto bad_y = y;
  bad_y[0] = nan;
  Throws([&] { model.Fit({x.data(), 8, 1}, bad_y); });
  x[0] = std::numeric_limits<float>::infinity();
  Throws([&] { model.Fit({x.data(), 8, 1}, y); });
  Throws([&] { model.Predict({x.data(), 8, 1}); });
  x[0] = nan;
  x[1] = nan;
  model.Fit({x.data(), 8, 1}, y);
  prediction = model.Predict({x.data(), 8, 1});
  for (float value : prediction) Check(std::isfinite(value));
  Check(prediction[0] < 0.01);
  // A missing-only split must work even when every observed value is identical.
  std::vector<float> missing_x{nan, nan, 1, 1};
  model.Fit({missing_x.data(), 4, 1}, {0, 0, 10, 10});
  prediction = model.Predict({missing_x.data(), 4, 1});
  Check(prediction[0] < 0.01 && prediction[2] > 9.99);
  std::vector<float> all_missing(4, nan);
  model.Fit({all_missing.data(), 4, 1}, {1, 2, 3, 4});
  Check(model.Predict({all_missing.data(), 4, 1})[0] == 2.5f);
  p.objective = libtree::Objective::kBinaryLogistic;
  Gbdt classifier(p);
  x = {0, 1, 2, 3, 4, 5, 6, 7};
  classifier.Fit({x.data(), 8, 1}, {0, 0, 0, 0, 1, 1, 1, 1});
  prediction = classifier.Predict({x.data(), 8, 1});
  Check(prediction.front() < 0.01 && prediction.back() > 0.99);
  Throws([&] { classifier.Fit({x.data(), 8, 1}, y); });
  classifier.Fit({x.data(), 8, 1}, std::vector<float>(8, 1));
  Check(classifier.Predict({x.data(), 8, 1})[0] > 0.99);
  // Extreme finite values cannot overflow threshold computation.
  x = {-3e38f, -3e38f, 3e38f, 3e38f};
  model.Fit({x.data(), 4, 1}, {0, 0, 10, 10});
  Check(model.Predict({x.data(), 4, 1})[0] < 0.01);
  for (int field = 0; field < 8; ++field) {
    Parameters invalid;
    if (field == 0) invalid.num_trees = 0;
    if (field == 1) invalid.max_depth = 31;
    if (field == 2) invalid.max_bins = 1;
    if (field == 3) invalid.min_samples_leaf = 0;
    if (field == 4) invalid.learning_rate = nan;
    if (field == 5) invalid.l2 = -1;
    if (field == 6) invalid.min_gain = -1;
    if (field == 7) invalid.objective = static_cast<libtree::Objective>(99);
    Throws([&] { Gbdt rejected(invalid); });
  }
  // Independent exhaustive SSE oracle for a single unregularized stump.
  // Enumerate partitions and evaluate residuals directly, without gradient
  // math.
  std::mt19937 generator(42);
  std::uniform_real_distribution<float> uniform(-5, 5);
  for (int trial = 0; trial < 30; ++trial) {
    std::vector<float> inputs(32), targets(32);
    for (int i = 0; i < 32; ++i) {
      inputs[i] = static_cast<float>(i);
      targets[i] = uniform(generator);
    }
    double best_error = std::numeric_limits<double>::infinity();
    int best_split = -1;
    double best_left = 0, best_right = 0;
    for (int split = 3; split <= 29; ++split) {
      const double left =
          std::accumulate(targets.begin(), targets.begin() + split, 0.0) /
          split;
      const double right =
          std::accumulate(targets.begin() + split, targets.end(), 0.0) /
          (32 - split);
      double error = 0;
      for (int i = 0; i < 32; ++i) {
        const double difference = targets[i] - (i < split ? left : right);
        error += difference * difference;
      }
      if (error < best_error) {
        best_error = error;
        best_split = split;
        best_left = left;
        best_right = right;
      }
    }
    Parameters stump;
    stump.num_trees = 1;
    stump.max_depth = 1;
    stump.min_samples_leaf = 3;
    stump.learning_rate = 1;
    stump.l2 = 0;
    Gbdt tested(stump);
    tested.Fit({inputs.data(), 32, 1}, targets);
    auto result = tested.Predict({inputs.data(), 32, 1});
    for (int i = 0; i < 32; ++i) {
      Check(std::abs(result[i] - (i < best_split ? best_left : best_right)) <
            1e-5);
    }
  }
  // Strong exception guarantee, zero-depth, minimum leaf and gain controls.
  Parameters constant;
  constant.max_depth = 0;
  Gbdt root_only(constant);
  x = {0, 1, 2, 3};
  root_only.Fit({x.data(), 4, 1}, {0, 0, 10, 10});
  Check(root_only.Predict({x.data(), 4, 1}) == std::vector<float>(4, 5));
  Throws([&] { root_only.Fit({x.data(), 4, 1}, {1}); });
  Check(root_only.Predict({x.data(), 4, 1}) == std::vector<float>(4, 5));
  Throws([&] { root_only.Predict({nullptr, 4, 1}); });
  Throws([&] { root_only.Fit({x.data(), 0, 0}, {}); });
  Throws([&] {
    root_only.Fit({x.data(), std::numeric_limits<std::size_t>::max(), 2}, {});
  });
  constant.max_depth = 4;
  constant.min_samples_leaf = 3;
  Gbdt min_leaf(constant);
  min_leaf.Fit({x.data(), 4, 1}, {0, 0, 10, 10});
  Check(min_leaf.Predict({x.data(), 4, 1}) == std::vector<float>(4, 5));
  constant.min_samples_leaf = 1;
  constant.min_gain = 1e9;
  Gbdt min_gain(constant);
  min_gain.Fit({x.data(), 4, 1}, {0, 0, 10, 10});
  Check(min_gain.Predict({x.data(), 4, 1}) == std::vector<float>(4, 5));
  // Representative workload: multi-feature split traversal and repeated refits.
  constexpr int rows = 1000, columns = 8;
  std::vector<float> dense(rows * columns), target(rows);
  for (float& value : dense) value = uniform(generator);
  for (int i = 0; i < rows; ++i)
    target[i] = dense[i * columns] + dense[i * columns + 1] * 2;
  Parameters workload;
  workload.num_trees = 30;
  workload.max_bins = 32;
  Gbdt dense_model(workload);
  dense_model.Fit({dense.data(), rows, columns}, target);
  auto result = dense_model.Predict({dense.data(), rows, columns});
  double mse = 0;
  for (int i = 0; i < rows; ++i)
    mse += std::pow(result[i] - target[i], 2) / rows;
  Check(mse < 1.5);
  dense_model.Fit({dense.data(), rows, columns}, target);
  Check(result == dense_model.Predict({dense.data(), rows, columns}));
  std::stringstream serialized;
  dense_model.SaveModel(serialized);
  Gbdt loaded;
  loaded.LoadModel(serialized);
  Check(loaded.Predict({dense.data(), rows, columns}) == result);
  CheckSerializedPrediction(loaded, {dense.data(), rows, columns});
  // Mixed sparse indicators, continuous columns, missing values and uneven
  // batches exercise histogram shortcuts and padded shallow trees.
  constexpr int mixed_rows = 137, mixed_columns = 12;
  std::vector<float> mixed(mixed_rows * mixed_columns), mixed_y(mixed_rows);
  for (int i = 0; i < mixed_rows; ++i) {
    for (int f = 0; f < mixed_columns; ++f)
      mixed[i * mixed_columns + f] =
          f < 9 ? static_cast<float>(i % 17 == f) : uniform(generator);
    if (i % 11 == 0) mixed[i * mixed_columns + 1] = nan;
    mixed_y[i] = mixed[i * mixed_columns] * 5 + mixed[i * mixed_columns + 9];
  }
  for (int depth : {0, 1, 4, 8, 9, 12}) {
    Parameters varied;
    varied.num_trees = 7;
    varied.max_depth = depth;
    varied.min_samples_leaf = 1;
    varied.l2 = 0;
    Gbdt varied_model(varied);
    varied_model.Fit({mixed.data(), mixed_rows, mixed_columns}, mixed_y);
    CheckSerializedPrediction(varied_model,
                              {mixed.data(), mixed_rows, mixed_columns});
    std::stringstream saved;
    varied_model.SaveModel(saved);
    Gbdt reloaded;
    reloaded.LoadModel(saved);
    CheckSerializedPrediction(reloaded,
                              {mixed.data(), mixed_rows, mixed_columns});
  }
  for (float& value : mixed_y) value = value > 0;
  Parameters binary_workload;
  binary_workload.objective = libtree::Objective::kBinaryLogistic;
  Gbdt binary_model(binary_workload);
  binary_model.Fit({mixed.data(), mixed_rows, mixed_columns}, mixed_y);
  CheckSerializedPrediction(binary_model,
                            {mixed.data(), mixed_rows, mixed_columns});
  binary_model.PredictInto({nullptr, 0, mixed_columns}, nullptr);
  Throws([&] {
    binary_model.PredictInto({mixed.data(), mixed_rows, mixed_columns},
                             nullptr);
  });
  // Quantization boundaries and both wide-matrix histogram layouts.
  constexpr int wide_rows = 80;
  for (int width : {20, 820}) {
    std::vector<float> wide(wide_rows * width), wide_y(wide_rows);
    for (float& value : wide) value = uniform(generator);
    for (int i = 0; i < wide_rows; ++i)
      wide_y[i] = wide[i * width] + wide[i * width + 1] * 2;
    for (int bin_limit : {2, 3, 17, 64, 65535}) {
      Parameters boundary;
      boundary.num_trees = 3;
      boundary.max_bins = bin_limit;
      Gbdt wide_model(boundary);
      wide_model.Fit({wide.data(), wide_rows, static_cast<std::size_t>(width)},
                     wide_y);
      CheckSerializedPrediction(wide_model, {wide.data(), wide_rows,
                                             static_cast<std::size_t>(width)});
    }
  }
  // Exercise finite bin 65,535 rather than only the parameter boundary. A
  // wrapped bin ID would masquerade as NaN and change the unseen-NaN route.
  std::vector<float> maximum_bins(65535), maximum_targets(65535);
  for (int i = 0; i < 65535; ++i) {
    maximum_bins[i] = static_cast<float>(i);
    maximum_targets[i] = i < 32768 ? 0 : 10;
  }
  Parameters maximum;
  maximum.num_trees = 1;
  maximum.max_depth = 1;
  maximum.max_bins = 65535;
  maximum.min_samples_leaf = 1;
  maximum.learning_rate = 1;
  maximum.l2 = 0;
  Gbdt maximum_model(maximum);
  maximum_model.Fit({maximum_bins.data(), maximum_bins.size(), 1},
                    maximum_targets);
  std::vector<float> probes{nan, 0, 65534};
  auto maximum_prediction = maximum_model.Predict({probes.data(), 3, 1});
  Check(std::abs(maximum_prediction[0]) < 1e-5 &&
        std::abs(maximum_prediction[1]) < 1e-5 &&
        std::abs(maximum_prediction[2] - 10) < 1e-5);
  CheckSerializedPrediction(maximum_model, {probes.data(), 3, 1});
  for (const std::string text :
       {"garbage", "LIBTREE_GBDT 99", "LIBTREE_GBDT 1\n-1"}) {
    std::stringstream invalid(text);
    Throws([&] { loaded.LoadModel(invalid); });
    Check(loaded.Predict({dense.data(), rows, columns}) == result);
  }
  std::stringstream truncated(
      serialized.str().substr(0, serialized.str().size() / 2));
  Throws([&] { loaded.LoadModel(truncated); });
  std::stringstream unfitted_output;
  Gbdt unfitted;
  Throws([&] { unfitted.SaveModel(unfitted_output); });
}
}  // namespace
int main() {
  try {
    Run();
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << checks << " checks passed\n";
}
