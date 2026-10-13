#ifndef LIBTREE_GBDT_H_
#define LIBTREE_GBDT_H_

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace libtree {
namespace internal {
class ParallelExecutor;
}
enum class Objective { kSquaredError, kBinaryLogistic };
enum class DumpFormat { kText = 0, kJson = 1 };

struct DecisionStep {
  int node_id;
  int feature_index;
  float threshold;
  float feature_value;  // NaN means the learned missing direction was used.
  bool missing_left;
  int next_node_id;
};

struct TreeExplanation {
  std::size_t tree_index;
  std::vector<DecisionStep> path;
  int leaf_id;
  double leaf_value;
  double contribution;  // learning_rate * leaf_value, on the margin scale.
};

struct PredictionExplanation {
  Objective objective;
  double base_score;
  double learning_rate;
  std::vector<TreeExplanation> trees;
  double margin;
  float prediction;  // Same float32 value returned by Predict; P(class=1) for
                     // binary.
};

struct Parameters {
  int num_trees = 100;
  int max_depth = 4;
  int max_bins = 64;
  int min_samples_leaf = 5;
  double learning_rate = 0.1;
  double l2 = 1.0;
  double min_gain = 0.0;
  Objective objective = Objective::kSquaredError;
  int num_threads = 1;  // Upper bound, including the calling thread; 1..256.
};

// Non-owning, contiguous row-major float matrix. NaN denotes missing data.
// data must point to at least rows * cols floats; the view cannot check
// capacity.
struct MatrixView {
  const float* data;
  std::size_t rows;
  std::size_t cols;
};

class Gbdt {
 public:
  explicit Gbdt(Parameters parameters = {});
  // Replaces the fitted model. The caller retains ownership of all inputs.
  void Fit(MatrixView features, const std::vector<float>& labels);
  std::vector<float> Predict(MatrixView features) const;
  // Writes rows predictions to caller-owned storage. output may be null only
  // for an empty batch and must not overlap the feature matrix.
  void PredictInto(MatrixView features, float* output) const;
  // Versioned text format; failed loads preserve any existing fitted model.
  void SaveModel(std::ostream& output) const;
  void LoadModel(std::istream& input);
  // Read-only inspection, separate from the versioned persistence format.
  // tree_index=-1 exports all trees; other indices are zero-based.
  // Names are UTF-8 display labels in column order; empty means f0, f1, ...
  void DumpModel(std::ostream& output, DumpFormat format = DumpFormat::kText,
                 int tree_index = -1,
                 const std::vector<std::string>& feature_names = {}) const;
  // Exactly one sample; paths include learned NaN routing and leaf
  // contributions.
  PredictionExplanation ExplainPrediction(MatrixView sample) const;
  void ExplainPrediction(
      MatrixView sample, std::ostream& output,
      DumpFormat format = DumpFormat::kText,
      const std::vector<std::string>& feature_names = {}) const;
  // Runtime configuration, independent of the portable model format.
  // Like Fit, must not run concurrently with predictions on this model.
  void SetNumThreads(int threads);
  const std::vector<double>& training_loss() const { return training_loss_; }
  std::size_t num_trees() const { return trees_.size(); }

 private:
  struct Node {
    int feature = -1;
    int left = -1;
    int right = -1;
    float threshold = 0;
    double value = 0;
    bool missing_left = true;
  };
  using Tree = std::vector<Node>;
  struct PredictionSplit {
    std::uint32_t feature = 0;
    float threshold = 0;
  };
  struct PredictionTree {
    int depth = -1;  // -1 selects ordinary traversal for deep/irregular trees.
    std::vector<PredictionSplit> splits;
    std::vector<double> leaves;
  };
  std::vector<PredictionTree> CompilePredictionTrees(
      const std::vector<Tree>& trees) const;
  void ValidatePredictionInput(MatrixView features) const;
  void PredictUnchecked(MatrixView features, float* output) const;
  void PredictRange(MatrixView features, float* output, std::size_t begin,
                    std::size_t end) const;
  Parameters parameters_;
  std::size_t num_features_ = 0;
  double base_score_ = 0;
  std::vector<Tree> trees_;
  std::vector<PredictionTree> prediction_trees_;
  std::vector<double> training_loss_;
  std::shared_ptr<internal::ParallelExecutor> executor_;
};
}  // namespace libtree
#endif  // LIBTREE_GBDT_H_
