#include <cmath>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>

#include "libtree/gbdt.h"

namespace libtree {
namespace {
constexpr std::size_t kMaxModelNodes = 1000000;
constexpr std::size_t kMaxModelTrees = 10000;
void Require(bool condition) {
  if (!condition) throw std::invalid_argument("invalid or truncated model");
}
}  // namespace
void Gbdt::SaveModel(std::ostream& output) const {
  if (trees_.empty()) throw std::logic_error("model is not fitted");
  std::size_t count = 0;
  for (const Tree& tree : trees_) count += tree.size();
  Require(count <= kMaxModelNodes && trees_.size() <= kMaxModelTrees);
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  output << "LIBTREE_GBDT 1\n"
         << static_cast<int>(parameters_.objective) << ' '
         << parameters_.num_trees << ' ' << parameters_.max_depth << ' '
         << parameters_.max_bins << ' ' << parameters_.min_samples_leaf << ' '
         << parameters_.learning_rate << ' ' << parameters_.l2 << ' '
         << parameters_.min_gain << ' ' << num_features_ << ' ' << base_score_
         << ' ' << trees_.size() << '\n';
  for (const Tree& tree : trees_) {
    output << tree.size() << '\n';
    for (const Node& node : tree) {
      output << node.feature << ' ' << node.left << ' ' << node.right << ' '
             << node.threshold << ' ' << node.value << ' ' << node.missing_left
             << '\n';
    }
  }
  if (!output) throw std::runtime_error("model write failed");
}
void Gbdt::LoadModel(std::istream& input) {
  std::string magic;
  int version = 0, objective = -1;
  Require(static_cast<bool>(input >> magic >> version));
  Require(magic == "LIBTREE_GBDT" && version == 1);
  Parameters p;
  std::size_t features = 0, tree_count = 0;
  double base = 0;
  Require(static_cast<bool>(input >> objective >> p.num_trees >> p.max_depth >>
                            p.max_bins >> p.min_samples_leaf >>
                            p.learning_rate >> p.l2 >> p.min_gain >> features >>
                            base >> tree_count));
  Require(objective == 0 || objective == 1);
  p.objective = static_cast<Objective>(objective);
  Gbdt parsed(p);
  Require(
      features > 0 &&
      features <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
      std::isfinite(base) && tree_count > 0 && tree_count <= kMaxModelTrees &&
      tree_count == static_cast<std::size_t>(p.num_trees));
  std::size_t total = 0;
  for (std::size_t t = 0; t < tree_count; ++t) {
    std::size_t count = 0;
    Require(static_cast<bool>(input >> count));
    Require(count > 0 && count <= kMaxModelNodes - total &&
            count <= (std::size_t{1} << (p.max_depth + 1)) - 1);
    total += count;
    Tree tree(count);
    std::vector<int> parent_count(count, 0), depth(count, 0);
    for (std::size_t i = 0; i < count; ++i) {
      Node& node = tree[i];
      int missing = -1;
      Require(static_cast<bool>(input >> node.feature >> node.left >>
                                node.right >> node.threshold >> node.value >>
                                missing));
      Require(std::isfinite(node.threshold) && std::isfinite(node.value) &&
              (missing == 0 || missing == 1));
      node.missing_left = missing == 1;
      if (node.feature == -1) {
        Require(node.left == -1 && node.right == -1);
      } else {
        Require(node.feature >= 0 &&
                static_cast<std::size_t>(node.feature) < features &&
                node.left > static_cast<int>(i) &&
                node.right > static_cast<int>(i) && node.left != node.right &&
                static_cast<std::size_t>(node.left) < count &&
                static_cast<std::size_t>(node.right) < count &&
                depth[i] < p.max_depth);
        Require(++parent_count[node.left] == 1 &&
                ++parent_count[node.right] == 1);
        depth[node.left] = depth[node.right] = depth[i] + 1;
      }
      if (i > 0) Require(parent_count[i] == 1);
    }
    parsed.trees_.push_back(std::move(tree));
  }
  input >> std::ws;
  Require(input.eof());
  parsed.num_features_ = features;
  parsed.base_score_ = base;
  parsed.prediction_trees_ = parsed.CompilePredictionTrees(parsed.trees_);
  *this = std::move(parsed);
}
}  // namespace libtree
