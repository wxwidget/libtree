#ifndef LIBTREE_C5_H_
#define LIBTREE_C5_H_

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "libtree/gbdt.h"

namespace libtree {
enum class SplitCriterion {
  kGainRatio = 0,
  kEntropy = 1,
  kGini = 2,
  kHellinger = 3
};
struct C5Parameters {
  int num_estimators = 10;
  int max_depth = 6;
  int max_bins = 64;
  int min_samples_leaf = 5;
  double learning_rate = 1;
  double pruning_alpha = 0.001;  // Global normalized error per extra leaf.
  double min_gain = 1e-9;
  SplitCriterion criterion = SplitCriterion::kGainRatio;
  int num_threads = 1;
};
// Finite values satisfy lower < x <= upper. NaN has a separate membership bit.
struct RuleCondition {
  int feature_index = 0;
  bool has_lower = false, has_upper = false, accepts_missing = true;
  float lower = 0, upper = 0;
};
struct ClassificationRule {
  std::size_t rule_id = 0, estimator_index = 0, support = 0;
  int class_index = 0;
  double purity = 0, vote = 0;
  std::vector<RuleCondition> conditions;
};
struct ClassificationStep {
  std::size_t node_id = 0, next_node_id = 0;
  int feature_index = 0;
  float value = 0, threshold = 0;  // NaN value means missing.
  bool missing_left = true, took_left = true;
};
struct ClassificationPath {
  std::size_t estimator_index = 0, leaf_id = 0, rule_id = 0;
  int class_index = 0;
  double vote = 0;
  std::vector<ClassificationStep> steps;
};
struct RuleExplanation {
  int prediction = 0;
  std::vector<double> votes, probabilities;
  std::vector<ClassificationRule> matched_rules;
  std::vector<ClassificationPath> paths;
};
// C5-inspired histogram trees + discrete multiclass SAMME, not C5.0 compatible.
// Numeric features/NaN only. Labels are contiguous integers 0..K-1 (K<=256).
// Predict/Dump/Explain may run concurrently; Fit must have exclusive access.
class C5Classifier {
 public:
  explicit C5Classifier(C5Parameters parameters = {});
  // Failed fits preserve the old model. Empty weights means uniform weights.
  void Fit(MatrixView x, const std::vector<int>& y,
           const std::vector<double>& sample_weight = {});
  std::vector<int> Predict(MatrixView x) const;
  // Output capacity is x.rows integers; output must not overlap x.
  void PredictInto(MatrixView x, int* output) const;
  // Row-major rows * num_classes doubles, normalized vote softmax,
  // uncalibrated.
  std::vector<double> PredictProba(MatrixView x) const;
  void PredictProbaInto(MatrixView x, double* output) const;
  const std::vector<ClassificationRule>& rules() const;
  const std::vector<double>& estimator_errors() const;
  std::size_t num_classes() const { return num_classes_; }
  RuleExplanation Explain(MatrixView sample) const;
  // Retained, pruned trees; tree_index=-1 means all estimators.
  void DumpModel(std::ostream& output, DumpFormat format = DumpFormat::kText,
                 int tree_index = -1,
                 const std::vector<std::string>& feature_names = {}) const;
  void DumpRules(std::ostream& output, DumpFormat format = DumpFormat::kText,
                 const std::vector<std::string>& feature_names = {}) const;
  void Explain(MatrixView sample, std::ostream& output,
               DumpFormat format = DumpFormat::kText,
               const std::vector<std::string>& feature_names = {}) const;

 private:
  struct Node {
    int feature = -1, left = -1, right = -1, label = 0;
    float threshold = 0;
    bool missing_left = true;
    std::size_t support = 0, rule_id = 0;
    double purity = 0;
  };
  struct Learner {
    std::vector<Node> nodes;
    double vote = 0;
  };
  class Builder;
  void Train(MatrixView x, const std::vector<int>& y,
             const std::vector<double>& weights);
  void Validate(MatrixView x) const;
  void RequireFitted() const;
  std::size_t Leaf(const Learner& tree, const float* row) const;
  void PredictRows(MatrixView x, double* probabilities, int* labels) const;
  void NormalizeVotes(const double* votes, double* probabilities) const;
  void ExtractRules(Learner* tree, std::size_t estimator);
  C5Parameters parameters_;
  std::size_t num_features_ = 0, num_classes_ = 0;
  std::vector<Learner> learners_;
  std::vector<ClassificationRule> rules_;
  std::vector<double> errors_;
  std::shared_ptr<internal::ParallelExecutor> executor_;
};
}  // namespace libtree
#endif  // LIBTREE_C5_H_
