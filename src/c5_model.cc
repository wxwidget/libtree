#include <cmath>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>

#include "libtree/c5.h"
#include "parallel.h"

namespace libtree {
namespace {
constexpr std::size_t kMaxNodes = 1000000;
void Require(bool value) {
  if (!value) throw std::invalid_argument("invalid or truncated C5 model");
}
}  // namespace
void C5Classifier::SaveModel(std::ostream& output) const {
  RequireFitted();
  std::size_t nodes = 0;
  for (const auto& tree : learners_) nodes += tree.nodes.size();
  Require(nodes <= kMaxNodes);
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  output << "LIBTREE_C5 1\n"
         << parameters_.num_estimators << ' ' << parameters_.max_depth << ' '
         << parameters_.max_bins << ' ' << parameters_.min_samples_leaf << ' '
         << parameters_.learning_rate << ' ' << parameters_.pruning_alpha << ' '
         << parameters_.min_gain << ' '
         << static_cast<int>(parameters_.criterion) << ' ' << num_features_
         << ' ' << num_classes_ << ' ' << learners_.size() << ' '
         << static_cast<bool>(executor_) << '\n';
  for (std::size_t t = 0; t < learners_.size(); ++t) {
    const auto& tree = learners_[t];
    output << tree.vote << ' ' << errors_[t] << ' ' << tree.nodes.size()
           << '\n';
    for (const auto& node : tree.nodes)
      output << node.feature << ' ' << node.left << ' ' << node.right << ' '
             << node.label << ' ' << node.threshold << ' ' << node.missing_left
             << ' ' << node.support << ' ' << node.purity << '\n';
  }
  if (!output) throw std::runtime_error("C5 model write failed");
}
void C5Classifier::LoadModel(std::istream& input) {
  std::string magic;
  int version = 0, criterion = -1, parallel = -1;
  Require(static_cast<bool>(input >> magic >> version));
  Require(magic == "LIBTREE_C5" && version == 1);
  C5Parameters p;
  std::size_t features = 0, classes = 0, trees = 0;
  Require(static_cast<bool>(
      input >> p.num_estimators >> p.max_depth >> p.max_bins >>
      p.min_samples_leaf >> p.learning_rate >> p.pruning_alpha >> p.min_gain >>
      criterion >> features >> classes >> trees >> parallel));
  p.criterion = static_cast<SplitCriterion>(criterion);
  p.num_threads = parameters_.num_threads;
  C5Classifier parsed(p);
  Require(features > 0 && features <= 100000 && classes > 0 && classes <= 256 &&
          trees > 0 && trees <= static_cast<std::size_t>(p.num_estimators) &&
          (parallel == 0 || parallel == 1) &&
          (p.criterion != SplitCriterion::kHellinger || classes <= 2));
  parsed.num_features_ = features;
  parsed.num_classes_ = classes;
  std::size_t total = 0;
  double total_vote = 0;
  for (std::size_t t = 0; t < trees; ++t) {
    Learner tree;
    double error = 0;
    std::size_t count = 0;
    Require(static_cast<bool>(input >> tree.vote >> error >> count));
    Require(std::isfinite(tree.vote) && tree.vote > 0 && std::isfinite(error) &&
            error >= 0 && error <= 1 + 1e-12 && count > 0 &&
            count <= kMaxNodes - total);
    total_vote += tree.vote;
    // Leave headroom when prediction regroups these positive votes by class.
    Require(total_vote <= std::numeric_limits<double>::max() / 2);
    total += count;
    tree.nodes.resize(count);
    std::vector<int> parents(count, 0), depth(count, 0);
    for (std::size_t i = 0; i < count; ++i) {
      auto& node = tree.nodes[i];
      int missing = -1;
      Require(static_cast<bool>(input >> node.feature >> node.left >>
                                node.right >> node.label >> node.threshold >>
                                missing >> node.support >> node.purity));
      Require(
          node.label >= 0 && static_cast<std::size_t>(node.label) < classes &&
          std::isfinite(node.threshold) && (missing == 0 || missing == 1) &&
          node.support > 0 && node.support <= 100000000 &&
          std::isfinite(node.purity) && node.purity >= 0 && node.purity <= 1);
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
        Require(++parents[node.left] == 1 && ++parents[node.right] == 1);
        depth[node.left] = depth[node.right] = depth[i] + 1;
      }
      // Pruning leaves unreachable subtrees in storage. Their nodes still obey
      // forward-edge/depth bounds, so rebuilding reachable rules is safe.
    }
    parsed.ExtractRules(&tree, t);
    parsed.learners_.push_back(std::move(tree));
    parsed.errors_.push_back(error);
  }
  input >> std::ws;
  Require(input.eof());
  if (parallel && p.num_threads > 1)
    parsed.executor_ =
        std::make_shared<internal::ParallelExecutor>(p.num_threads);
  *this = std::move(parsed);
}
}  // namespace libtree
