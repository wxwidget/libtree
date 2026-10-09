#ifndef LIBTREE_GBDT_H_
#define LIBTREE_GBDT_H_

#include <cstddef>
#include <iosfwd>
#include <vector>

namespace libtree {
enum class Objective { kSquaredError, kBinaryLogistic };

struct Parameters {
  int num_trees = 100;
  int max_depth = 4;
  int max_bins = 64;
  int min_samples_leaf = 5;
  double learning_rate = 0.1;
  double l2 = 1.0;
  double min_gain = 0.0;
  Objective objective = Objective::kSquaredError;
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
  // Versioned text format; failed loads preserve any existing fitted model.
  void SaveModel(std::ostream& output) const;
  void LoadModel(std::istream& input);
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
  Parameters parameters_;
  std::size_t num_features_ = 0;
  double base_score_ = 0;
  std::vector<Tree> trees_;
  std::vector<double> training_loss_;
};
}  // namespace libtree
#endif  // LIBTREE_GBDT_H_
