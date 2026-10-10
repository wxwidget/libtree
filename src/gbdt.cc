#include "libtree/gbdt.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "parallel.h"

namespace libtree {
namespace {
double Sigmoid(double value) {
  if (value >= 0) return 1 / (1 + std::exp(-value));
  const double exp_value = std::exp(value);
  return exp_value / (1 + exp_value);
}
void ValidateShape(MatrixView matrix, bool allow_empty) {
  if (matrix.cols == 0 || (!allow_empty && matrix.rows == 0) ||
      (matrix.rows != 0 && matrix.data == nullptr) ||
      matrix.rows > std::numeric_limits<std::size_t>::max() / matrix.cols) {
    throw std::invalid_argument("invalid matrix shape or null data");
  }
}
void ValidateMatrix(MatrixView matrix, bool allow_empty) {
  ValidateShape(matrix, allow_empty);
  for (std::size_t i = 0; i < matrix.rows * matrix.cols; ++i) {
    if (std::isinf(matrix.data[i])) {
      throw std::invalid_argument("features must be finite or NaN");
    }
  }
}
struct Derivative {
  double gradient;
  double hessian;
};
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
  Stats& operator+=(const Derivative& other) {
    gradient += other.gradient;
    hessian += other.hessian;
    ++count;
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
struct SplitCandidate {
  int feature = -1;
  std::size_t bin = 0;
  bool missing_left = true;
  Stats left;
  Stats right;
};
struct SplitSearchConfig {
  std::size_t min_samples_leaf;
  double l2;
  double min_gain;
};
SplitCandidate FindBestSplit(
    const Stats& total, const Stats* histograms,
    const std::vector<std::size_t>& offsets,
    const std::vector<std::vector<float>>& cuts,
    const SplitSearchConfig& config) {
  SplitCandidate best;
  double best_gain = config.min_gain;
  const double parent_score = Score(total, config.l2);
  for (std::size_t feature = 0; feature < cuts.size(); ++feature) {
    const std::size_t num_bins = cuts[feature].size();
    if (num_bins == 0) continue;
    const auto* histogram = histograms + offsets[feature];
    Stats prefix;
    for (std::size_t bin = 1; bin <= num_bins; ++bin) {
      prefix += histogram[bin];
      for (bool missing_left : {true, false}) {
        // Without missing rows both directions describe the same partition.
        if (!missing_left && histogram[0].count == 0) continue;
        const Stats left = missing_left ? prefix + histogram[0] : prefix;
        const Stats right = total - left;
        if (left.count < config.min_samples_leaf ||
            right.count < config.min_samples_leaf)
          continue;
        const double gain =
            0.5 * (Score(left, config.l2) + Score(right, config.l2) -
                   parent_score);
        if (gain > best_gain + 1e-12) {
          best_gain = gain;
          best.feature = static_cast<int>(feature);
          best.bin = bin;
          best.missing_left = missing_left;
          best.left = left;
          best.right = right;
        }
      }
    }
  }
  return best;
}
}  // namespace

Gbdt::Gbdt(Parameters parameters) : parameters_(parameters) {
  if (parameters.num_trees < 1 || parameters.max_depth < 0 ||
      parameters.max_depth > 20 || parameters.max_bins < 2 ||
      parameters.max_bins > 65535 || parameters.min_samples_leaf < 1 ||
      parameters.num_threads < 1 || parameters.num_threads > 256 ||
      !std::isfinite(parameters.learning_rate) ||
      parameters.learning_rate <= 0 || parameters.learning_rate > 1 ||
      !std::isfinite(parameters.l2) || parameters.l2 < 0 ||
      !std::isfinite(parameters.min_gain) || parameters.min_gain < 0 ||
      (parameters.objective != Objective::kSquaredError &&
       parameters.objective != Objective::kBinaryLogistic)) {
    throw std::invalid_argument("invalid GBDT parameters");
  }
}

void Gbdt::SetNumThreads(int threads) {
  if (threads < 1 || threads > 256)
    throw std::invalid_argument("num_threads must be 1..256");
  if (threads == parameters_.num_threads && (threads == 1 || executor_)) return;
  auto executor = threads == 1
                      ? nullptr
                      : std::make_shared<internal::ParallelExecutor>(threads);
  parameters_.num_threads = threads;
  executor_ = std::move(executor);
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
  auto executor = executor_;
  if (!executor && parameters_.num_threads > 1)
    executor =
        std::make_shared<internal::ParallelExecutor>(parameters_.num_threads);
  auto for_features = [&](std::size_t work, auto function,
                          bool allow_parallel = true) {
    if (allow_parallel && executor && d > 1 && work >= 65536)
      executor->For(d, function);
    else
      for (std::size_t f = 0; f < d; ++f) function(0, f);
  };
  std::vector<std::vector<float>> cuts(d);
  // Quantize once per Fit. Column-major uint16 bins keep histogram scans
  // compact.
  std::vector<std::uint16_t> bins(n * d);
  struct SparseColumn {
    std::uint16_t default_bin = 0;
    int single_bin = -1;
    std::vector<std::size_t> exceptions;
    bool enabled = false;
  };
  std::vector<SparseColumn> sparse_columns(d);
  std::vector<std::vector<float>> sort_workspace(parameters_.num_threads);
  for_features(n * d, [&](int worker, std::size_t f) {
    auto& sorted = sort_workspace[worker];
    sorted.reserve(n);
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
    // A padded search table permits fixed-step, branch-free bin lookup while
    // preserving the exact original cuts (including repeated-value quantiles).
    std::size_t padded = 1;
    while (padded < cuts[f].size()) padded *= 2;
    sorted.resize(padded);
    if (!cuts[f].empty()) {
      std::copy(cuts[f].begin(), cuts[f].end(), sorted.begin());
      std::fill(sorted.begin() + cuts[f].size(), sorted.end(), cuts[f].back());
    }
    for (std::size_t i = 0; i < n; ++i) {
      const float value = features.data[i * d + f];
      if (std::isnan(value)) {
        bins[f * n + i] = 0;
      } else {
        std::size_t lower = 0;
        for (std::size_t step = padded / 2; step; step /= 2)
          lower += (sorted[lower + step - 1] < value) * step;
        bins[f * n + i] = static_cast<std::uint16_t>(lower + 1);
      }
    }
    std::vector<std::size_t> counts(cuts[f].size() + 1, 0);
    for (std::size_t i = 0; i < n; ++i) ++counts[bins[f * n + i]];
    auto& sparse = sparse_columns[f];
    sparse.default_bin = static_cast<std::uint16_t>(
        std::max_element(counts.begin(), counts.end()) - counts.begin());
    if (n - counts[sparse.default_bin] <= n / 2) {
      sparse.enabled = true;
      for (std::size_t i = 0; i < n; ++i)
        if (bins[f * n + i] != sparse.default_bin)
          sparse.exceptions.push_back(i);
      for (std::size_t b = 0; b < counts.size(); ++b)
        if (b != sparse.default_bin && counts[b]) {
          if (sparse.single_bin >= 0) {
            sparse.single_bin = -1;
            break;
          }
          sparse.single_bin = static_cast<int>(b);
        }
    }
  });
  std::vector<Tree> trees;
  trees.reserve(parameters_.num_trees);
  std::vector<double> losses;
  losses.reserve(parameters_.num_trees);
  std::vector<double> margins(n, base);
  std::vector<Derivative> derivatives(n), ordered_derivatives;
  // Gradient and loss updates are independent per row. Keep fixed-size chunks
  // so their reduction order is stable across thread counts while exposing a
  // useful amount of work after every tree.
  constexpr std::size_t kDerivativeChunk = 4096;
  std::vector<double> derivative_losses(
      (n + kDerivativeChunk - 1) / kDerivativeChunk);
  const double initial_prediction = binary ? Sigmoid(base) : base;
  const double initial_hessian =
      binary ? std::max(initial_prediction * (1 - initial_prediction), 1e-6)
             : 1;
  for (std::size_t i = 0; i < n; ++i)
    derivatives[i] = {initial_prediction - labels[i], initial_hessian};
  std::vector<std::size_t> rows(n);
  std::vector<std::size_t> positions;
  std::vector<std::size_t> offsets(d + 1, 0);
  for (std::size_t f = 0; f < d; ++f)
    offsets[f + 1] = offsets[f] + cuts[f].size() + 1;
  // Wide matrices benefit from scanning a row once and reusing its gradient
  // for all features. Store absolute histogram indices to remove offset loads
  // in this hot loop. Larger histograms retain the column-oriented fallback.
  std::vector<std::uint16_t> row_histogram_indices;
  const std::size_t continuous = std::count_if(
      cuts.begin(), cuts.end(), [](const auto& cut) { return cut.size() > 4; });
  if (d >= 16 && continuous > d / 2 && offsets.back() <= 65536) {
    row_histogram_indices.resize(n * d);
    for_features(n * d, [&](int, std::size_t f) {
      for (std::size_t row = 0; row < n; ++row)
        row_histogram_indices[row * d + f] =
            static_cast<std::uint16_t>(offsets[f] + bins[f * n + row]);
    });
  }
  if (row_histogram_indices.empty()) {
    ordered_derivatives.resize(n);
    positions.resize(n);
  }
  // Reuse only a depth-sized workspace: build the smaller child, then obtain
  // its sibling by subtracting from the parent histogram.
  std::vector<std::vector<Stats>> workspace(parameters_.max_depth + 1);
  const SplitSearchConfig split_config{
      static_cast<std::size_t>(parameters_.min_samples_leaf), parameters_.l2,
      parameters_.min_gain};
  auto sum_rows = [&](std::size_t begin, std::size_t end) {
    Stats total;
    for (std::size_t j = begin; j < end; ++j) total += derivatives[rows[j]];
    return total;
  };
  auto make_histograms = [&](std::size_t begin, std::size_t end,
                             const Stats& total, Stats* histograms,
                             const Stats* parent = nullptr,
                             bool allow_parallel = true) {
    std::fill(histograms, histograms + offsets.back(), Stats{});
    if (!row_histogram_indices.empty()) {
      if (allow_parallel && executor && (end - begin) * d >= 65536) {
        // Disjoint feature histograms, with identical per-bin row order.
        // Unlike row-wise reductions, this is bit-identical across threads.
        const std::size_t blocks = parameters_.num_threads;
        executor->For(blocks, [&](int, std::size_t block) {
          const std::size_t first = d * block / blocks;
          const std::size_t last = d * (block + 1) / blocks;
          for (std::size_t j = begin; j < end; ++j) {
            const std::size_t row = rows[j];
            const auto* indices = row_histogram_indices.data() + row * d;
            const Derivative derivative = derivatives[row];
            for (std::size_t f = first; f < last; ++f)
              histograms[indices[f]] += derivative;
          }
        });
        return;
      }
      for (std::size_t j = begin; j < end; ++j) {
        const std::size_t row = rows[j];
        const auto* indices = row_histogram_indices.data() + row * d;
        const Derivative derivative = derivatives[row];
        for (std::size_t f = 0; f < d; ++f)
          histograms[indices[f]] += derivative;
      }
      return;
    }
    for (std::size_t j = begin; j < end; ++j)
      ordered_derivatives[j] = derivatives[rows[j]];
    for_features((end - begin) * d, [&](int, std::size_t f) {
      const std::size_t num_bins = cuts[f].size();
      if (num_bins == 0) return;
      auto* histogram = histograms + offsets[f];
      if (parent && num_bins <= 4) {
        int occupied = -1;
        for (std::size_t b = 0; b <= num_bins; ++b)
          if (parent[offsets[f] + b].count) {
            if (occupied >= 0) {
              occupied = -1;
              break;
            }
            occupied = static_cast<int>(b);
          }
        if (occupied >= 0) {
          histogram[occupied] = total;
          return;
        }
      }
      const auto* column = bins.data() + f * n;
      const auto& sparse = sparse_columns[f];
      const bool root = begin == 0 && end == n;
      if (sparse.enabled &&
          (root || sparse.exceptions.size() < (end - begin) / 2)) {
        if (sparse.single_bin >= 0 && root) {
          std::array<Stats, 4> partial{};
          std::size_t j = 0;
          for (; j + 4 <= sparse.exceptions.size(); j += 4)
            for (std::size_t lane = 0; lane < 4; ++lane)
              partial[lane] += derivatives[sparse.exceptions[j + lane]];
          for (; j < sparse.exceptions.size(); ++j)
            partial[0] += derivatives[sparse.exceptions[j]];
          for (const auto& lane : partial) histogram[sparse.single_bin] += lane;
        } else {
          for (std::size_t row : sparse.exceptions) {
            if (root || (positions[row] >= begin && positions[row] < end))
              histogram[column[row]] += derivatives[row];
          }
        }
        Stats other;
        for (std::size_t b = 0; b <= num_bins; ++b)
          if (b != sparse.default_bin) other += histogram[b];
        histogram[sparse.default_bin] = total - other;
      } else if (sparse.single_bin >= 0) {
        // Two-valued columns need only one weighted sum. Masking avoids
        // scattered histogram writes and unpredictable per-row branches.
        std::array<Stats, 4> partial{};
        std::size_t j = begin;
        for (; j + 4 <= end; j += 4)
          for (std::size_t lane = 0; lane < 4; ++lane) {
            const std::size_t row = rows[j + lane];
            const std::size_t hit = column[row] == sparse.single_bin;
            partial[lane].gradient +=
                ordered_derivatives[j + lane].gradient * hit;
            partial[lane].hessian +=
                ordered_derivatives[j + lane].hessian * hit;
            partial[lane].count += hit;
          }
        for (; j < end; ++j) {
          const std::size_t row = rows[j];
          const std::size_t hit = column[row] == sparse.single_bin;
          partial[0].gradient += ordered_derivatives[j].gradient * hit;
          partial[0].hessian += ordered_derivatives[j].hessian * hit;
          partial[0].count += hit;
        }
        Stats other;
        for (const auto& lane : partial) other += lane;
        histogram[sparse.single_bin] = other;
        histogram[sparse.default_bin] = total - other;
      } else if (num_bins <= 4) {
        // Independent accumulators avoid a serial dependency chain for
        // binary/low-cardinality columns dominated by one bin.
        std::array<std::array<Stats, 5>, 4> partial{};
        std::size_t j = begin;
        for (; j + 4 <= end; j += 4)
          for (std::size_t lane = 0; lane < 4; ++lane) {
            const std::size_t row = rows[j + lane];
            partial[lane][column[row]] += ordered_derivatives[j + lane];
          }
        for (; j < end; ++j) {
          const std::size_t row = rows[j];
          partial[0][column[row]] += ordered_derivatives[j];
        }
        for (std::size_t b = 0; b <= num_bins; ++b)
          for (const auto& lane : partial) histogram[b] += lane[b];
      } else {
        for (std::size_t j = begin; j < end; ++j) {
          const std::size_t row = rows[j];
          histogram[column[row]] += ordered_derivatives[j];
        }
      }
    }, allow_parallel);
  };
  for (int iteration = 0; iteration < parameters_.num_trees; ++iteration) {
    std::iota(rows.begin(), rows.end(), 0);
    std::iota(positions.begin(), positions.end(), 0);
    const Stats root_total = sum_rows(0, n);
    if (parameters_.max_depth > 0 &&
        n / 2 >= static_cast<std::size_t>(parameters_.min_samples_leaf)) {
      workspace[0].resize(offsets.back());
      make_histograms(0, n, root_total, workspace[0].data());
    }
    Tree tree;
    tree.reserve(std::min<std::size_t>(
        2 * n, (std::size_t{1} << (parameters_.max_depth + 1)) - 1));
    auto build = [&](auto&& self, std::size_t begin, std::size_t end, int depth,
                     Stats* histograms, const Stats& total) -> int {
      const int node_id = static_cast<int>(tree.size());
      Node node;
      node.value = -total.gradient / (total.hessian + parameters_.l2);
      tree.push_back(node);
      auto finish_leaf = [&] {
        const double update = parameters_.learning_rate * node.value;
        for (std::size_t j = begin; j < end; ++j) margins[rows[j]] += update;
        return node_id;
      };
      if (depth >= parameters_.max_depth ||
          total.count / 2 <
              static_cast<std::size_t>(parameters_.min_samples_leaf))
        return finish_leaf();
      const SplitCandidate split =
          FindBestSplit(total, histograms, offsets, cuts, split_config);
      if (split.feature < 0) return finish_leaf();
      const auto* column =
          bins.data() + static_cast<std::size_t>(split.feature) * n;
      auto mid = std::partition(
          rows.begin() + begin, rows.begin() + end, [&](std::size_t row) {
            return column[row] == 0 ? split.missing_left
                                    : column[row] <= split.bin;
          });
      const std::size_t middle = mid - rows.begin();
      if (!positions.empty())
        for (std::size_t j = begin; j < end; ++j) positions[rows[j]] = j;
      tree[node_id].feature = split.feature;
      tree[node_id].threshold = cuts[split.feature][split.bin - 1];
      tree[node_id].missing_left = split.missing_left;
      int left, right;
      const Stats left_total = split.left;
      const Stats right_total = split.right;
      if (depth + 1 == parameters_.max_depth) {
        left = self(self, begin, middle, depth + 1, nullptr, left_total);
        right = self(self, middle, end, depth + 1, nullptr, right_total);
      } else {
        auto& small_histogram = workspace[depth + 1];
        small_histogram.resize(offsets.back());
        const bool left_small = middle - begin <= end - middle;
        const std::size_t small_begin = left_small ? begin : middle;
        const std::size_t small_end = left_small ? middle : end;
        const Stats small_total = left_small ? left_total : right_total;
        make_histograms(small_begin, small_end, small_total,
                        small_histogram.data(), histograms);
        for (std::size_t b = 0; b < offsets.back(); ++b)
          histograms[b] = histograms[b] - small_histogram[b];
        // No reference into tree survives recursion (vector may reallocate).
        const int small = self(self, small_begin, small_end, depth + 1,
                               small_histogram.data(), small_total);
        const std::size_t large_begin = left_small ? middle : begin;
        const std::size_t large_end = left_small ? end : middle;
        const int large = self(self, large_begin, large_end, depth + 1,
                               histograms, left_small ? right_total : left_total);
        left = left_small ? small : large;
        right = left_small ? large : small;
      }
      tree[node_id].left = left;
      tree[node_id].right = right;
      return node_id;
    };
    const std::size_t kLayerMemoryLimit = 64 * 1024 * 1024;
    const bool layer_parallel =
        executor && parameters_.max_depth <= 12 && n * d >= 262144 &&
        offsets.back() <=
            kLayerMemoryLimit /
                (sizeof(Stats) * (std::size_t{1} << parameters_.max_depth));
    if (!layer_parallel) {
      build(build, 0, n, 0, workspace[0].data(), root_total);
    } else {
      struct LayerNode {
        std::size_t begin = 0, end = 0, middle = 0;
        Stats total, left_total, right_total;
        int tree_index = -1, feature = -1;
        std::size_t bin = 0;
        bool missing_left = true;
        std::vector<Stats> histograms;
      };
      const std::size_t no_child = std::numeric_limits<std::size_t>::max();
      std::vector<LayerNode> level;
      std::vector<std::uint8_t> right_child_first(1, 0);
      LayerNode root;
      root.begin = 0;
      root.end = n;
      root.total = root_total;
      root.tree_index = 0;
      root.histograms = std::move(workspace[0]);
      level.push_back(std::move(root));
      tree.resize(1);
      for (int depth = 0; depth < parameters_.max_depth && !level.empty();
           ++depth) {
        auto evaluate = [&](std::size_t index) {
          auto& current = level[index];
          Node& output = tree[current.tree_index];
          output.value =
              -current.total.gradient / (current.total.hessian + parameters_.l2);
          if (current.total.count / 2 <
              static_cast<std::size_t>(parameters_.min_samples_leaf)) {
            for (std::size_t j = current.begin; j < current.end; ++j)
              margins[rows[j]] += parameters_.learning_rate * output.value;
            return;
          }
          const SplitCandidate split = FindBestSplit(
              current.total, current.histograms.data(), offsets, cuts,
              split_config);
          current.feature = split.feature;
          if (split.feature < 0) {
            for (std::size_t j = current.begin; j < current.end; ++j)
              margins[rows[j]] += parameters_.learning_rate * output.value;
            return;
          }
          current.bin = split.bin;
          current.missing_left = split.missing_left;
          current.left_total = split.left;
          current.right_total = split.right;
          const auto* column =
              bins.data() + static_cast<std::size_t>(split.feature) * n;
          const auto middle = std::partition(
              rows.begin() + current.begin, rows.begin() + current.end,
              [&](std::size_t row) {
                return column[row] == 0 ? split.missing_left
                                        : column[row] <= split.bin;
              });
          current.middle = static_cast<std::size_t>(middle - rows.begin());
          if (!positions.empty())
            for (std::size_t j = current.begin; j < current.end; ++j)
              positions[rows[j]] = j;
          output.feature = split.feature;
          output.threshold = cuts[split.feature][split.bin - 1];
          output.missing_left = split.missing_left;
        };
        if (level.size() > 1) {
          executor->For(level.size(), [&](int, std::size_t index) {
            evaluate(index);
          });
        } else {
          evaluate(0);
        }

        std::vector<std::array<std::size_t, 2>> child_indices(
            level.size(), {no_child, no_child});
        std::size_t child_count = 0;
        for (std::size_t i = 0; i < level.size(); ++i) {
          if (level[i].feature >= 0) {
            child_indices[i] = {child_count, child_count + 1};
            child_count += 2;
          }
        }
        if (child_count == 0) break;
        const std::size_t first_tree_child = tree.size();
        tree.resize(first_tree_child + child_count);
        right_child_first.resize(tree.size(), 0);
        const bool has_next_histograms = depth + 1 < parameters_.max_depth;
        std::vector<LayerNode> next(has_next_histograms ? child_count : 0);
        for (std::size_t i = 0; i < level.size(); ++i) {
          const auto child = child_indices[i];
          if (child[0] == no_child) continue;
          auto& parent = level[i];
          auto& output = tree[parent.tree_index];
          const std::size_t left_index = first_tree_child + child[0];
          const std::size_t right_index = first_tree_child + child[1];
          output.left = static_cast<int>(left_index);
          output.right = static_cast<int>(right_index);
          const std::size_t left_begin = parent.begin;
          const std::size_t left_end = parent.middle;
          const std::size_t right_begin = parent.middle;
          const std::size_t right_end = parent.end;
          const Stats left_total = parent.left_total;
          const Stats right_total = parent.right_total;
          const bool left_small = left_end - left_begin <=
                                  right_end - right_begin;
          right_child_first[parent.tree_index] =
              has_next_histograms && !left_small;
          if (!has_next_histograms) {
            tree[left_index].value =
                -left_total.gradient / (left_total.hessian + parameters_.l2);
            tree[right_index].value =
                -right_total.gradient / (right_total.hessian + parameters_.l2);
            for (std::size_t j = left_begin; j < left_end; ++j)
              margins[rows[j]] +=
                  parameters_.learning_rate * tree[left_index].value;
            for (std::size_t j = right_begin; j < right_end; ++j)
              margins[rows[j]] +=
                  parameters_.learning_rate * tree[right_index].value;
            continue;
          }
          auto& left_node = next[child[0]];
          auto& right_node = next[child[1]];
          left_node.begin = left_begin;
          left_node.end = left_end;
          left_node.total = left_total;
          left_node.tree_index = static_cast<int>(left_index);
          right_node.begin = right_begin;
          right_node.end = right_end;
          right_node.total = right_total;
          right_node.tree_index = static_cast<int>(right_index);
        }
        if (has_next_histograms) {
          auto prepare_children = [&](std::size_t index,
                                      bool allow_inner_parallel) {
            const auto child = child_indices[index];
            if (child[0] == no_child) return;
            auto& parent = level[index];
            auto& left_node = next[child[0]];
            auto& right_node = next[child[1]];
            const bool left_small = left_node.end - left_node.begin <=
                                    right_node.end - right_node.begin;
            auto& small = left_small ? left_node : right_node;
            auto& large = left_small ? right_node : left_node;
            small.histograms.resize(offsets.back());
            make_histograms(small.begin, small.end, small.total,
                            small.histograms.data(), parent.histograms.data(),
                            allow_inner_parallel);
            for (std::size_t b = 0; b < offsets.back(); ++b)
              parent.histograms[b] =
                  parent.histograms[b] - small.histograms[b];
            large.histograms = std::move(parent.histograms);
          };
          if (level.size() > 1) {
            executor->For(level.size(), [&](int, std::size_t index) {
              prepare_children(index, false);
            });
          } else {
            prepare_children(0, true);
          }
          level = std::move(next);
        }
      }

      // Restore the established depth-first node order so serialized models
      // remain byte-identical to the recursive builder.
      Tree depth_first;
      depth_first.reserve(tree.size());
      auto append_depth_first = [&](auto&& self, int old_index) -> int {
        const int new_index = static_cast<int>(depth_first.size());
        depth_first.push_back(tree[old_index]);
        if (tree[old_index].feature >= 0) {
          int left, right;
          if (right_child_first[old_index]) {
            right = self(self, tree[old_index].right);
            left = self(self, tree[old_index].left);
          } else {
            left = self(self, tree[old_index].left);
            right = self(self, tree[old_index].right);
          }
          depth_first[new_index].left = left;
          depth_first[new_index].right = right;
        }
        return new_index;
      };
      append_depth_first(append_depth_first, 0);
      tree = std::move(depth_first);
    }
    auto update_derivatives = [&](std::size_t chunk) {
      const std::size_t begin = chunk * kDerivativeChunk;
      const std::size_t end = std::min(n, begin + kDerivativeChunk);
      double partial_loss = 0;
      for (std::size_t i = begin; i < end; ++i) {
        if (binary) {
          // Stable softplus cross entropy without log(0) or exp overflow.
          // Reuse its exponential to prepare the next boosting iteration.
          const double e = std::exp(-std::abs(margins[i]));
          const double prediction =
              margins[i] >= 0 ? 1 / (1 + e) : e / (1 + e);
          partial_loss += std::max(margins[i], 0.0) -
                          labels[i] * margins[i] + std::log1p(e);
          derivatives[i] = {prediction - labels[i],
                            std::max(prediction * (1 - prediction), 1e-6)};
        } else {
          const double residual = margins[i] - labels[i];
          partial_loss += residual * residual;
          derivatives[i] = {residual, 1};
        }
      }
      derivative_losses[chunk] = partial_loss;
    };
    if (executor && derivative_losses.size() > 1) {
      executor->For(derivative_losses.size(), [&](int, std::size_t chunk) {
        update_derivatives(chunk);
      });
    } else {
      for (std::size_t chunk = 0; chunk < derivative_losses.size(); ++chunk)
        update_derivatives(chunk);
    }
    double loss = 0;
    for (double partial_loss : derivative_losses) loss += partial_loss;
    losses.push_back(loss / n);
    trees.push_back(std::move(tree));
  }
  auto prediction_trees = CompilePredictionTrees(trees);
  num_features_ = d;
  base_score_ = base;
  trees_ = std::move(trees);
  prediction_trees_ = std::move(prediction_trees);
  training_loss_ = std::move(losses);
  executor_ = std::move(executor);
}

std::vector<Gbdt::PredictionTree> Gbdt::CompilePredictionTrees(
    const std::vector<Tree>& trees) const {
  constexpr std::uint32_t kMissingLeft = std::uint32_t{1} << 31;
  std::vector<PredictionTree> compiled;
  compiled.reserve(trees.size());
  for (const auto& tree : trees) {
    std::vector<int> depths(tree.size(), 0);
    int max_depth = 0;
    for (std::size_t i = 0; i < tree.size(); ++i) {
      max_depth = std::max(max_depth, depths[i]);
      if (tree[i].feature >= 0)
        depths[tree[i].left] = depths[tree[i].right] = depths[i] + 1;
    }
    PredictionTree result;
    // Bound padding by both depth and original storage; a thin deep tree uses
    // the compact ordinary representation rather than exponential expansion.
    const std::size_t leaves = std::size_t{1} << max_depth;
    if (max_depth <= 8 && (leaves * 16 - 8) <= tree.size() * sizeof(Node)) {
      result.depth = max_depth;
      result.splits.resize(leaves - 1);
      result.leaves.resize(leaves);
      auto expand = [&](auto&& self, int index, std::size_t heap,
                        int depth) -> void {
        const Node& node = tree[index];
        if (node.feature < 0) {
          const std::size_t length = std::size_t{1} << (max_depth - depth);
          const std::size_t first =
              (heap + 1 - (std::size_t{1} << depth)) * length;
          std::fill_n(result.leaves.begin() + first, length,
                      parameters_.learning_rate * node.value);
          return;
        }
        result.splits[heap] = {static_cast<std::uint32_t>(node.feature) |
                                   (node.missing_left ? kMissingLeft : 0),
                               node.threshold};
        self(self, node.left, heap * 2 + 1, depth + 1);
        self(self, node.right, heap * 2 + 2, depth + 1);
      };
      expand(expand, 0, 0, 0);
    }
    compiled.push_back(std::move(result));
  }
  return compiled;
}  // GCOVR_EXCL_LINE: compiler emits a non-executable closing brace counter.

void Gbdt::ValidatePredictionInput(MatrixView features) const {
  if (trees_.empty()) throw std::logic_error("model is not fitted");
  if (features.cols != num_features_)
    throw std::invalid_argument("feature count mismatch");
  if (executor_) {
    ValidateShape(features, true);
    const std::size_t count = features.rows * features.cols;
    if (count >= 65536) {
      constexpr std::size_t kChunk = 4096;
      executor_->For((count + kChunk - 1) / kChunk, [&](int,
                                                        std::size_t chunk) {
        const std::size_t begin = chunk * kChunk;
        for (std::size_t i = begin; i < std::min(count, begin + kChunk); ++i)
          if (std::isinf(features.data[i]))
            throw std::invalid_argument("features must be finite or NaN");
      });
      return;
    }
  }
  ValidateMatrix(features, true);
}

std::vector<float> Gbdt::Predict(MatrixView features) const {
  ValidatePredictionInput(features);
  std::vector<float> prediction(features.rows);
  PredictUnchecked(features, prediction.data());
  return prediction;
}  // GCOVR_EXCL_LINE: compiler emits a non-executable closing brace counter.

void Gbdt::PredictInto(MatrixView features, float* output) const {
  ValidatePredictionInput(features);
  if (features.rows && output == nullptr)
    throw std::invalid_argument("null prediction output");
  PredictUnchecked(features, output);
}

void Gbdt::PredictUnchecked(MatrixView features, float* output) const {
  constexpr std::size_t kBatch = 64;
  if (executor_ && features.rows >= 512 && trees_.size() >= 8) {
    executor_->For((features.rows + kBatch - 1) / kBatch,
                   [&](int, std::size_t batch) {
                     const std::size_t begin = batch * kBatch;
                     PredictRange(features, output, begin,
                                  std::min(features.rows, begin + kBatch));
                   });
  } else {
    PredictRange(features, output, 0, features.rows);
  }
}

void Gbdt::PredictRange(MatrixView features, float* output, std::size_t first,
                        std::size_t last) const {
  constexpr std::size_t kBatch = 64;
  constexpr std::uint32_t kMissingLeft = std::uint32_t{1} << 31;
  for (std::size_t begin = first; begin < last; begin += kBatch) {
    const std::size_t count = std::min(kBatch, last - begin);
    std::array<double, kBatch> margins;
    std::array<const float*, kBatch> inputs;
    std::fill_n(margins.begin(), count, base_score_);
    for (std::size_t row = 0; row < count; ++row)
      inputs[row] = features.data + (begin + row) * features.cols;
    for (std::size_t t = 0; t < trees_.size(); ++t) {
      const auto& compiled = prediction_trees_[t];
      if (compiled.depth >= 0) {
        std::array<std::uint32_t, kBatch> indices;
        std::fill_n(indices.begin(), count, 0);
        for (int depth = 0; depth < compiled.depth; ++depth)
          for (std::size_t row = 0; row < count; ++row) {
            const auto& split = compiled.splits[indices[row]];
            const float value = inputs[row][split.feature & ~kMissingLeft];
            const bool right =
                (value > split.threshold) |
                (std::isnan(value) && !(split.feature & kMissingLeft));
            indices[row] = indices[row] * 2 + 1 + right;
          }
        for (std::size_t row = 0; row < count; ++row)
          margins[row] +=
              compiled.leaves[indices[row] - compiled.splits.size()];
      } else {
        const auto& tree = trees_[t];
        for (std::size_t row = 0; row < count; ++row) {
          int index = 0;
          while (tree[index].feature >= 0) {
            const Node& node = tree[index];
            const float value = inputs[row][node.feature];
            const bool left =
                std::isnan(value) ? node.missing_left : value <= node.threshold;
            index = left ? node.left : node.right;
          }
          margins[row] += parameters_.learning_rate * tree[index].value;
        }
      }
    }
    for (std::size_t row = 0; row < count; ++row)
      output[begin + row] =
          static_cast<float>(parameters_.objective == Objective::kBinaryLogistic
                                 ? Sigmoid(margins[row])
                                 : margins[row]);
  }
}
}  // namespace libtree
