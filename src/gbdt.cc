#include "libtree/gbdt.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace libtree {
namespace {
double Sigmoid(double value) {
  if (value >= 0) return 1 / (1 + std::exp(-value));
  const double exp_value = std::exp(value);
  return exp_value / (1 + exp_value);
}
void ValidateMatrix(MatrixView matrix, bool allow_empty) {
  if (matrix.cols == 0 || (!allow_empty && matrix.rows == 0) ||
      (matrix.rows != 0 && matrix.data == nullptr) ||
      matrix.rows > std::numeric_limits<std::size_t>::max() / matrix.cols) {
    throw std::invalid_argument("invalid matrix shape or null data");
  }
  for (std::size_t i = 0; i < matrix.rows * matrix.cols; ++i) {
    if (std::isinf(matrix.data[i])) {
      throw std::invalid_argument("features must be finite or NaN");
    }
  }
}
struct Stats {
  double gradient = 0;
  double hessian = 0;
  std::size_t count = 0;
  Stats& operator+=(const Stats& other) {
    gradient += other.gradient;
    hessian += other.hessian;
    count += other.count;
    return *this;
  }
};
Stats operator+(Stats a, const Stats& b) { return a += b; }
Stats operator-(const Stats& a, const Stats& b) {
  return {a.gradient - b.gradient, a.hessian - b.hessian, a.count - b.count};
}
double Score(const Stats& stats, double l2) {
  return stats.gradient * stats.gradient / (stats.hessian + l2);
}
}  // namespace

Gbdt::Gbdt(Parameters parameters) : parameters_(parameters) {
  if (parameters.num_trees < 1 || parameters.max_depth < 0 ||
      parameters.max_depth > 20 || parameters.max_bins < 2 ||
      parameters.max_bins > 65535 || parameters.min_samples_leaf < 1 ||
      !std::isfinite(parameters.learning_rate) ||
      parameters.learning_rate <= 0 || parameters.learning_rate > 1 ||
      !std::isfinite(parameters.l2) || parameters.l2 < 0 ||
      !std::isfinite(parameters.min_gain) || parameters.min_gain < 0 ||
      (parameters.objective != Objective::kSquaredError &&
       parameters.objective != Objective::kBinaryLogistic)) {
    throw std::invalid_argument("invalid GBDT parameters");
  }
}

void Gbdt::Fit(MatrixView features, const std::vector<float>& labels) {
  ValidateMatrix(features, false);
  if (features.cols >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      labels.size() != features.rows) {
    throw std::invalid_argument("labels or feature count mismatch");
  }
  const bool binary = parameters_.objective == Objective::kBinaryLogistic;
  double mean = 0;
  for (float label : labels) {
    if (!std::isfinite(label) || (binary && label != 0 && label != 1)) {
      throw std::invalid_argument("invalid label for objective");
    }
    mean += label;
  }
  mean /= labels.size();
  // Fit into local state: exceptions leave any previously fitted model intact.
  double base = mean;
  if (binary) {
    mean = std::clamp(mean, 1e-6, 1 - 1e-6);
    base = std::log(mean / (1 - mean));
  }
  const std::size_t n = features.rows;
  const std::size_t d = features.cols;
  std::vector<std::vector<float>> cuts(d);
  // Quantize once per Fit. Column-major uint16 bins keep histogram scans
  // compact.
  std::vector<std::uint16_t> bins(n * d);
  std::vector<float> sorted;
  sorted.reserve(n);
  for (std::size_t f = 0; f < d; ++f) {
    sorted.clear();
    for (std::size_t i = 0; i < n; ++i) {
      const float value = features.data[i * d + f];
      if (!std::isnan(value)) sorted.push_back(value);
    }
    std::sort(sorted.begin(), sorted.end());
    if (!sorted.empty()) {
      const std::size_t count = std::min(
          sorted.size(), static_cast<std::size_t>(parameters_.max_bins));
      for (std::size_t k = 1; k <= count; ++k) {
        const float cut = sorted[(k * sorted.size() - 1) / count];
        if (cuts[f].empty() || cut > cuts[f].back()) cuts[f].push_back(cut);
      }
    }
    for (std::size_t i = 0; i < n; ++i) {
      const float value = features.data[i * d + f];
      bins[f * n + i] =
          std::isnan(value)
              ? 0
              : static_cast<std::uint16_t>(
                    std::lower_bound(cuts[f].begin(), cuts[f].end(), value) -
                    cuts[f].begin() + 1);
    }
  }
  std::vector<Tree> trees;
  trees.reserve(parameters_.num_trees);
  std::vector<double> losses;
  losses.reserve(parameters_.num_trees);
  std::vector<double> margins(n, base);
  std::vector<Stats> derivatives(n);
  std::vector<std::size_t> rows(n);
  std::vector<Stats> histogram(parameters_.max_bins + 1);
  for (int iteration = 0; iteration < parameters_.num_trees; ++iteration) {
    for (std::size_t i = 0; i < n; ++i) {
      const double prediction = binary ? Sigmoid(margins[i]) : margins[i];
      derivatives[i] = {
          prediction - labels[i],
          binary ? std::max(prediction * (1 - prediction), 1e-6) : 1, 1};
    }
    std::iota(rows.begin(), rows.end(), 0);
    Tree tree;
    tree.reserve(std::min<std::size_t>(
        2 * n, (std::size_t{1} << (parameters_.max_depth + 1)) - 1));
    std::function<int(std::size_t, std::size_t, int)> build;
    build = [&](std::size_t begin, std::size_t end, int depth) -> int {
      Stats total;
      for (std::size_t j = begin; j < end; ++j) total += derivatives[rows[j]];
      const int node_id = static_cast<int>(tree.size());
      Node node;
      node.value = -total.gradient / (total.hessian + parameters_.l2);
      tree.push_back(node);
      if (depth >= parameters_.max_depth ||
          total.count / 2 <
              static_cast<std::size_t>(parameters_.min_samples_leaf))
        return node_id;
      double best_gain = parameters_.min_gain;
      const double parent_score = Score(total, parameters_.l2);
      int best_feature = -1;
      std::size_t best_bin = 0;
      bool best_missing_left = true;
      for (std::size_t f = 0; f < d; ++f) {
        const std::size_t num_bins = cuts[f].size();
        if (num_bins == 0) continue;
        std::fill(histogram.begin(), histogram.begin() + num_bins + 1, Stats{});
        const auto* column = bins.data() + f * n;
        for (std::size_t j = begin; j < end; ++j) {
          const std::size_t row = rows[j];
          histogram[column[row]] += derivatives[row];
        }
        Stats prefix;
        for (std::size_t k = 1; k <= num_bins; ++k) {
          prefix += histogram[k];
          for (bool missing_left : {true, false}) {
            // Without missing rows both directions describe the same partition.
            if (!missing_left && histogram[0].count == 0) continue;
            const Stats left = missing_left ? prefix + histogram[0] : prefix;
            const Stats right = total - left;
            if (left.count <
                    static_cast<std::size_t>(parameters_.min_samples_leaf) ||
                right.count <
                    static_cast<std::size_t>(parameters_.min_samples_leaf))
              continue;
            const double gain =
                0.5 * (Score(left, parameters_.l2) +
                       Score(right, parameters_.l2) - parent_score);
            if (gain > best_gain + 1e-12) {
              best_gain = gain;
              best_feature = static_cast<int>(f);
              best_bin = k;
              best_missing_left = missing_left;
            }
          }
        }
      }
      if (best_feature < 0) return node_id;
      const auto* column =
          bins.data() + static_cast<std::size_t>(best_feature) * n;
      auto mid = std::partition(
          rows.begin() + begin, rows.begin() + end, [&](std::size_t row) {
            return column[row] == 0 ? best_missing_left
                                    : column[row] <= best_bin;
          });
      const std::size_t middle = mid - rows.begin();
      tree[node_id].feature = best_feature;
      tree[node_id].threshold = cuts[best_feature][best_bin - 1];
      tree[node_id].missing_left = best_missing_left;
      // No reference into tree survives recursion (vector may reallocate).
      const int left = build(begin, middle, depth + 1);
      const int right = build(middle, end, depth + 1);
      tree[node_id].left = left;
      tree[node_id].right = right;
      return node_id;
    };
    build(0, n, 0);
    double loss = 0;
    for (std::size_t i = 0; i < n; ++i) {
      int index = 0;
      while (tree[index].feature >= 0) {
        const Node& node = tree[index];
        const float value = features.data[i * d + node.feature];
        const bool left =
            std::isnan(value) ? node.missing_left : value <= node.threshold;
        index = left ? node.left : node.right;
      }
      margins[i] += parameters_.learning_rate * tree[index].value;
      if (binary) {
        // Stable softplus cross entropy without log(0) or exp overflow.
        loss += std::max(margins[i], 0.0) - labels[i] * margins[i] +
                std::log1p(std::exp(-std::abs(margins[i])));
      } else {
        const double residual = margins[i] - labels[i];
        loss += residual * residual;
      }
    }
    losses.push_back(loss / n);
    trees.push_back(std::move(tree));
  }
  num_features_ = d;
  base_score_ = base;
  trees_ = std::move(trees);
  training_loss_ = std::move(losses);
}

std::vector<float> Gbdt::Predict(MatrixView features) const {
  if (trees_.empty()) throw std::logic_error("model is not fitted");
  if (features.cols != num_features_)
    throw std::invalid_argument("feature count mismatch");
  ValidateMatrix(features, true);
  std::vector<float> prediction(features.rows, 0);
  for (std::size_t i = 0; i < features.rows; ++i) {
    double margin = base_score_;
    for (const Tree& tree : trees_) {
      int index = 0;
      while (tree[index].feature >= 0) {
        const Node& node = tree[index];
        const float value = features.data[i * features.cols + node.feature];
        const bool left =
            std::isnan(value) ? node.missing_left : value <= node.threshold;
        index = left ? node.left : node.right;
      }
      margin += parameters_.learning_rate * tree[index].value;
    }
    prediction[i] = static_cast<float>(
        parameters_.objective == Objective::kBinaryLogistic ? Sigmoid(margin)
                                                            : margin);
  }
  return prediction;
}
}  // namespace libtree
