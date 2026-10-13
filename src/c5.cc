#include "libtree/c5.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "parallel.h"

namespace libtree {
namespace {
constexpr std::size_t kParallelWork = 65536;
void CheckMatrix(MatrixView x) {
  if (x.cols == 0 || x.cols > 100000 || x.rows > 100000000 ||
      x.rows > std::numeric_limits<std::size_t>::max() / x.cols ||
      (x.rows != 0 && x.data == nullptr))
    throw std::invalid_argument("invalid feature matrix");
  for (std::size_t i = 0; i < x.rows * x.cols; ++i)
    if (std::isinf(x.data[i]))
      throw std::invalid_argument("infinite feature value");
}
double Impurity(const std::vector<double>& counts, double weight,
                SplitCriterion criterion) {
  if (weight <= 0) return 0;
  double result = criterion == SplitCriterion::kGini ? 1 : 0;
  for (double count : counts) {
    const double p = count / weight;
    if (p <= 0) continue;
    if (criterion == SplitCriterion::kGini)
      result -= p * p;
    else
      result -= p * std::log2(p);
  }
  return result;
}
int Winner(const std::vector<double>& counts) {
  return static_cast<int>(std::max_element(counts.begin(), counts.end()) -
                          counts.begin());
}
}  // namespace

// Each feature owns a histogram task. Reduction is in feature order, ensuring
// bit-for-bit thread-count independence. Quantiles are built once per fit.
class C5Classifier::Builder {
 public:
  Builder(C5Classifier* model, MatrixView x, const std::vector<int>& y)
      : model_(model), x_(x), y_(y), cuts_(x.cols), bins_(x.rows * x.cols) {
    const auto build = [&](int, std::size_t f) {
      std::vector<float> values;
      for (std::size_t r = 0; r < x.rows; ++r)
        if (!std::isnan(x.data[r * x.cols + f]))
          values.push_back(x.data[r * x.cols + f]);
      std::sort(values.begin(), values.end());
      auto& cuts = cuts_[f];
      for (int b = 1; b < model_->parameters_.max_bins && !values.empty();
           ++b) {
        const float v =
            values[values.size() * b / model_->parameters_.max_bins];
        if (v < values.back() && (cuts.empty() || v > cuts.back()))
          cuts.push_back(v);
      }
      // Missingness itself can be predictive even for constant finite values.
      if (!values.empty() && values.size() < x.rows)
        cuts.push_back(values.back());
      for (std::size_t r = 0; r < x.rows; ++r) {
        const float v = x.data[r * x.cols + f];
        bins_[f * x.rows + r] = static_cast<std::uint16_t>(
            std::isnan(v)
                ? cuts.size() + 1
                : std::lower_bound(cuts.begin(), cuts.end(), v) - cuts.begin());
      }
    };
    Dispatch(x.rows, build);
  }
  Learner Build(const std::vector<double>& weights) {
    weights_ = &weights;
    Learner tree;
    std::vector<std::size_t> rows(x_.rows);
    std::iota(rows.begin(), rows.end(), 0);
    Grow(&tree, rows, 0);
    // Compact after pruning; discarded subtrees occupy no inference memory.
    Learner compact;
    std::function<int(int)> copy = [&](int id) {
      const int target = static_cast<int>(compact.nodes.size());
      compact.nodes.push_back(tree.nodes[id]);
      if (tree.nodes[id].feature >= 0) {
        const int left = copy(tree.nodes[id].left);
        const int right = copy(tree.nodes[id].right);
        compact.nodes[target].left = left;
        compact.nodes[target].right = right;
      }
      return target;
    };
    copy(0);
    return compact;
  }

 private:
  struct Split {
    int feature = -1;
    float threshold = 0;
    bool missing_left = true;
    double gain = 0, score = 0;
  };
  struct Risk {
    double error;
    int leaves;
  };
  template <typename Function>
  void Dispatch(std::size_t rows, Function function) {
    if (model_->executor_ && rows * x_.cols >= kParallelWork)
      model_->executor_->For(x_.cols, function);
    else
      for (std::size_t f = 0; f < x_.cols; ++f) function(0, f);
  }
  Split FeatureSplit(const std::vector<std::size_t>& rows, std::size_t f,
                     const std::vector<double>& total, double weight) const {
    const std::size_t k = model_->num_classes_, regular = cuts_[f].size() + 1;
    std::vector<double> hist((regular + 1) * k), left(k), right(k),
        candidate(k);
    std::vector<std::size_t> counts(regular + 1);
    for (std::size_t r : rows) {
      const auto bin = bins_[f * x_.rows + r];
      hist[bin * k + y_[r]] += (*weights_)[r];
      ++counts[bin];
    }
    Split best;
    const bool hellinger =
        model_->parameters_.criterion == SplitCriterion::kHellinger;
    const double parent =
        hellinger ? 0 : Impurity(total, weight, model_->parameters_.criterion);
    std::size_t left_count = 0;
    for (std::size_t b = 0; b + 1 < regular; ++b) {
      left_count += counts[b];
      for (std::size_t c = 0; c < k; ++c) left[c] += hist[b * k + c];
      for (bool missing_left : {true, false}) {
        const std::size_t count =
            left_count + (missing_left ? counts[regular] : 0);
        if (count < static_cast<std::size_t>(
                        model_->parameters_.min_samples_leaf) ||
            rows.size() - count <
                static_cast<std::size_t>(model_->parameters_.min_samples_leaf))
          continue;
        for (std::size_t c = 0; c < k; ++c) {
          candidate[c] = left[c];
          if (missing_left) candidate[c] += hist[regular * k + c];
          right[c] = std::max(0., total[c] - candidate[c]);
        }
        const double lw =
            std::accumulate(candidate.begin(), candidate.end(), 0.);
        const double rw = std::accumulate(right.begin(), right.end(), 0.);
        if (lw <= 0 || rw <= 0) continue;
        const double p = lw / (lw + rw);
        double gain;
        // Cieslak et al., DMKD 2012, doi:10.1007/s10618-011-0222-1.
        // Class-conditional branch probabilities remove the class-prior factor.
        if (hellinger) {
          const double left_distance = std::sqrt(candidate[0] / total[0]) -
                                       std::sqrt(candidate[1] / total[1]);
          const double right_distance =
              std::sqrt(right[0] / total[0]) - std::sqrt(right[1] / total[1]);
          gain = std::hypot(left_distance, right_distance);
        } else {
          gain = parent -
                 p * Impurity(candidate, lw, model_->parameters_.criterion) -
                 (1 - p) * Impurity(right, rw, model_->parameters_.criterion);
        }
        const double score =
            model_->parameters_.criterion == SplitCriterion::kGainRatio
                ? gain / (-p * std::log2(p) - (1 - p) * std::log2(1 - p))
                : gain;
        if (gain > model_->parameters_.min_gain && score > best.score) {
          best.feature = static_cast<int>(f);
          best.threshold = cuts_[f][b];
          best.missing_left = missing_left;
          best.gain = gain;
          best.score = score;
        }
      }
    }
    return best;
  }
  Risk Grow(Learner* tree, const std::vector<std::size_t>& rows, int depth) {
    const int id = static_cast<int>(tree->nodes.size());
    std::vector<double> total(model_->num_classes_);
    for (std::size_t r : rows) total[y_[r]] += (*weights_)[r];
    const double weight = std::accumulate(total.begin(), total.end(), 0.);
    Node node;
    node.label = Winner(total);
    node.support = rows.size();
    node.purity = weight > 0 ? total[node.label] / weight : 0;
    tree->nodes.push_back(node);
    const double leaf_error = weight - total[node.label];
    if (depth >= model_->parameters_.max_depth || leaf_error <= 1e-15 ||
        rows.size() / 2 <
            static_cast<std::size_t>(model_->parameters_.min_samples_leaf))
      return {leaf_error, 1};
    std::vector<Split> splits(x_.cols);
    Dispatch(rows.size(), [&](int, std::size_t f) {
      splits[f] = FeatureSplit(rows, f, total, weight);
    });
    double mean_gain = 0;
    std::size_t candidates = 0;
    for (const auto& split : splits)
      if (split.feature >= 0) {
        mean_gain += split.gain;
        ++candidates;
      }
    mean_gain /= std::max<std::size_t>(candidates, 1);
    Split best;
    for (const auto& split : splits) {
      if (model_->parameters_.criterion == SplitCriterion::kGainRatio &&
          split.gain + 1e-15 < mean_gain)
        continue;
      if (split.score > best.score) best = split;
    }
    if (best.feature < 0) return {leaf_error, 1};
    std::vector<std::size_t> left, right;
    for (std::size_t r : rows) {
      const float v = x_.data[r * x_.cols + best.feature];
      ((std::isnan(v) ? best.missing_left : v <= best.threshold) ? left : right)
          .push_back(r);
    }
    const int left_id = static_cast<int>(tree->nodes.size());
    const Risk lr = Grow(tree, left, depth + 1);
    const int right_id = static_cast<int>(tree->nodes.size());
    const Risk rr = Grow(tree, right, depth + 1);
    const Risk subtree{lr.error + rr.error, lr.leaves + rr.leaves};
    const double alpha = model_->parameters_.pruning_alpha;
    if (alpha > 0 &&
        leaf_error + alpha <= subtree.error + alpha * subtree.leaves)
      return {leaf_error, 1};
    tree->nodes[id].feature = best.feature;
    tree->nodes[id].threshold = best.threshold;
    tree->nodes[id].missing_left = best.missing_left;
    tree->nodes[id].left = left_id;
    tree->nodes[id].right = right_id;
    return subtree;
  }
  C5Classifier* model_;
  MatrixView x_;
  const std::vector<int>& y_;
  std::vector<std::vector<float>> cuts_;
  std::vector<std::uint16_t> bins_;
  const std::vector<double>* weights_ = nullptr;
};

C5Classifier::C5Classifier(C5Parameters parameters) : parameters_(parameters) {
  const auto& p = parameters_;
  if (p.num_estimators < 1 || p.num_estimators > 1000 || p.max_depth < 0 ||
      p.max_depth > 32 || p.max_bins < 2 || p.max_bins > 256 ||
      p.min_samples_leaf < 1 || p.num_threads < 1 || p.num_threads > 256 ||
      !std::isfinite(p.learning_rate) || p.learning_rate <= 0 ||
      p.learning_rate > 1 || !std::isfinite(p.pruning_alpha) ||
      p.pruning_alpha < 0 || !std::isfinite(p.min_gain) || p.min_gain < 0 ||
      (p.criterion != SplitCriterion::kGainRatio &&
       p.criterion != SplitCriterion::kEntropy &&
       p.criterion != SplitCriterion::kGini &&
       p.criterion != SplitCriterion::kHellinger))
    throw std::invalid_argument("invalid C5 parameters");
}
void C5Classifier::Fit(MatrixView x, const std::vector<int>& y,
                       const std::vector<double>& sample_weight) {
  CheckMatrix(x);
  if (x.rows == 0 || y.size() != x.rows ||
      (!sample_weight.empty() && sample_weight.size() != x.rows))
    throw std::invalid_argument("labels/weights must match nonempty X");
  int largest = 0;
  for (int label : y) {
    if (label < 0 || label >= 256)
      throw std::invalid_argument("class index outside 0..255");
    largest = std::max(largest, label);
  }
  std::vector<bool> seen(largest + 1);
  for (int label : y) seen[label] = true;
  if (std::find(seen.begin(), seen.end(), false) != seen.end())
    throw std::invalid_argument("class indices must be contiguous from zero");
  if (parameters_.criterion == SplitCriterion::kHellinger && largest > 1)
    throw std::invalid_argument("Hellinger requires binary classification");
  std::vector<double> weights = sample_weight;
  if (weights.empty()) weights.assign(x.rows, 1);
  // Scale before summation to avoid overflow with large finite input weights.
  double maximum = 0;
  for (double weight : weights) {
    if (!std::isfinite(weight) || weight < 0)
      throw std::invalid_argument("invalid sample weight");
    maximum = std::max(maximum, weight);
  }
  if (maximum == 0) throw std::invalid_argument("all sample weights are zero");
  for (double& weight : weights) weight /= maximum;
  const double sum = std::accumulate(weights.begin(), weights.end(), 0.);
  for (double& weight : weights) weight /= sum;
  C5Classifier next(parameters_);
  next.num_features_ = x.cols;
  next.num_classes_ = largest + 1;
  if (parameters_.num_threads > 1 && x.rows * x.cols >= kParallelWork)
    next.executor_ =
        std::make_shared<internal::ParallelExecutor>(parameters_.num_threads);
  next.Train(x, y, weights);
  *this = std::move(next);
}
void C5Classifier::Train(MatrixView x, const std::vector<int>& y,
                         const std::vector<double>& initial_weights) {
  // Zhu et al., Multi-class AdaBoost (2009), doi:10.4310/SII.2009.v2.n3.a8.
  // Shrinkage learning_rate scales both learner votes and weight updates.
  Builder builder(this, x, y);
  auto weights = initial_weights;
  for (int round = 0; round < parameters_.num_estimators; ++round) {
    Learner tree = builder.Build(weights);
    std::vector<bool> wrong(x.rows);
    double error = 0;
    for (std::size_t r = 0; r < x.rows; ++r) {
      wrong[r] = tree.nodes[Leaf(tree, x.data + r * x.cols)].label != y[r];
      if (wrong[r]) error += weights[r];
    }
    const bool weak = error >= 1 - 1. / num_classes_ - 1e-15;
    if (weak && !learners_.empty()) break;
    // A first learner worse than chance falls back to its majority-class tree.
    tree.vote = weak || num_classes_ == 1
                    ? 1
                    : parameters_.learning_rate *
                          (std::log((1 - std::clamp(error, 1e-15, 1 - 1e-15)) /
                                    std::clamp(error, 1e-15, 1 - 1e-15)) +
                           std::log(num_classes_ - 1.));
    tree.vote = std::max(tree.vote, std::numeric_limits<double>::min());
    ExtractRules(&tree, learners_.size());
    learners_.push_back(std::move(tree));
    errors_.push_back(error);
    if (weak || error <= 1e-15 || round + 1 == parameters_.num_estimators)
      break;
    const double factor = std::exp(learners_.back().vote);
    for (std::size_t r = 0; r < x.rows; ++r)
      if (wrong[r]) weights[r] *= factor;
    const double sum = std::accumulate(weights.begin(), weights.end(), 0.);
    for (double& weight : weights) weight /= sum;
  }
}
std::size_t C5Classifier::Leaf(const Learner& tree, const float* row) const {
  int id = 0;
  while (tree.nodes[id].feature >= 0) {
    const auto& node = tree.nodes[id];
    const float v = row[node.feature];
    id = (std::isnan(v) ? node.missing_left : v <= node.threshold) ? node.left
                                                                   : node.right;
  }
  return static_cast<std::size_t>(id);
}
void C5Classifier::ExtractRules(Learner* tree, std::size_t estimator) {
  using Conditions = std::map<int, RuleCondition>;
  std::function<void(int, Conditions)> walk = [&](int id,
                                                  Conditions conditions) {
    auto& node = tree->nodes[id];
    if (node.feature < 0) {
      ClassificationRule rule;
      rule.rule_id = rules_.size();
      rule.estimator_index = estimator;
      rule.support = node.support;
      rule.class_index = node.label;
      rule.purity = node.purity;
      rule.vote = tree->vote;
      for (const auto& entry : conditions)
        rule.conditions.push_back(entry.second);
      node.rule_id = rule.rule_id;
      rules_.push_back(std::move(rule));
      return;
    }
    auto left = conditions;
    auto& lc = left[node.feature];
    lc.feature_index = node.feature;
    lc.upper =
        lc.has_upper ? std::min(lc.upper, node.threshold) : node.threshold;
    lc.has_upper = true;
    lc.accepts_missing = lc.accepts_missing && node.missing_left;
    walk(node.left, std::move(left));
    auto& rc = conditions[node.feature];
    rc.feature_index = node.feature;
    rc.lower =
        rc.has_lower ? std::max(rc.lower, node.threshold) : node.threshold;
    rc.has_lower = true;
    rc.accepts_missing = rc.accepts_missing && !node.missing_left;
    walk(node.right, std::move(conditions));
  };
  walk(0, {});
}
void C5Classifier::RequireFitted() const {
  if (learners_.empty()) throw std::logic_error("C5 model is not fitted");
}
void C5Classifier::Validate(MatrixView x) const {
  RequireFitted();
  CheckMatrix(x);
  if (x.cols != num_features_)
    throw std::invalid_argument("feature count mismatch");
}
void C5Classifier::FillVotes(const float* row,
                             std::vector<double>* votes) const {
  std::fill(votes->begin(), votes->end(), 0.);
  for (const auto& tree : learners_)
    (*votes)[tree.nodes[Leaf(tree, row)].label] += tree.vote;
}
std::vector<double> C5Classifier::Probabilities(
    const std::vector<double>& votes) const {
  const double sum = std::accumulate(votes.begin(), votes.end(), 0.);
  const double best = *std::max_element(votes.begin(), votes.end());
  std::vector<double> probability(num_classes_);
  for (std::size_t c = 0; c < num_classes_; ++c)
    probability[c] = std::exp((votes[c] - best) / sum);
  const double normalizer =
      std::accumulate(probability.begin(), probability.end(), 0.);
  for (double& value : probability) value /= normalizer;
  return probability;
}
std::vector<int> C5Classifier::Predict(MatrixView x) const {
  Validate(x);
  std::vector<int> output(x.rows);
  std::vector<std::vector<double>> scratch(
      executor_ ? parameters_.num_threads : 1,
      std::vector<double>(num_classes_));
  const auto predict = [&](int worker, std::size_t r) {
    FillVotes(x.data + r * x.cols, &scratch[worker]);
    output[r] = Winner(scratch[worker]);
  };
  if (executor_ && x.rows * learners_.size() >= kParallelWork)
    executor_->For(x.rows, predict);
  else
    for (std::size_t r = 0; r < x.rows; ++r) predict(0, r);
  return output;
}
std::vector<double> C5Classifier::PredictProba(MatrixView x) const {
  Validate(x);
  std::vector<double> output(x.rows * num_classes_);
  PredictProbaInto(x, output.data());
  return output;
}
void C5Classifier::PredictProbaInto(MatrixView x, double* output) const {
  Validate(x);
  if (x.rows != 0 && output == nullptr)
    throw std::invalid_argument("null probability output");
  std::vector<std::vector<double>> scratch(
      executor_ ? parameters_.num_threads : 1,
      std::vector<double>(num_classes_));
  const auto predict = [&](int worker, std::size_t r) {
    auto& votes = scratch[worker];
    FillVotes(x.data + r * x.cols, &votes);
    const double sum = std::accumulate(votes.begin(), votes.end(), 0.);
    const double best = *std::max_element(votes.begin(), votes.end());
    double* probability = output + r * num_classes_;
    double normalizer = 0;
    for (std::size_t c = 0; c < num_classes_; ++c) {
      probability[c] = std::exp((votes[c] - best) / sum);
      normalizer += probability[c];
    }
    for (std::size_t c = 0; c < num_classes_; ++c) probability[c] /= normalizer;
  };
  if (executor_ && x.rows * learners_.size() >= kParallelWork)
    executor_->For(x.rows, predict);
  else
    for (std::size_t r = 0; r < x.rows; ++r) predict(0, r);
}
const std::vector<ClassificationRule>& C5Classifier::rules() const {
  RequireFitted();
  return rules_;
}
const std::vector<double>& C5Classifier::estimator_errors() const {
  RequireFitted();
  return errors_;
}
RuleExplanation C5Classifier::Explain(MatrixView sample) const {
  Validate(sample);
  if (sample.rows != 1)
    throw std::invalid_argument("explanation requires one sample");
  RuleExplanation explanation;
  explanation.votes.resize(num_classes_);
  FillVotes(sample.data, &explanation.votes);
  explanation.prediction = Winner(explanation.votes);
  explanation.probabilities = Probabilities(explanation.votes);
  for (const auto& tree : learners_)
    explanation.matched_rules.push_back(
        rules_[tree.nodes[Leaf(tree, sample.data)].rule_id]);
  return explanation;
}
}  // namespace libtree
