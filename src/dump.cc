#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "inspection.h"
#include "libtree/gbdt.h"

namespace libtree {
namespace {
void ValidateFormat(DumpFormat format) {
  if (format != DumpFormat::kText && format != DumpFormat::kJson)
    throw std::invalid_argument("dump format must be text or json");
}
std::vector<std::string> FeatureNames(const std::vector<std::string>& names,
                                      std::size_t count) {
  if (!names.empty()) {
    if (names.size() != count)
      throw std::invalid_argument("feature name count mismatch");
    return names;
  }
  std::vector<std::string> result;
  result.reserve(count);
  for (std::size_t i = 0; i < count; ++i)
    result.push_back("f" + std::to_string(i));
  return result;
}
void Configure(std::ostream& output) {
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
}
void Metadata(std::ostream& output, DumpFormat format, Objective objective,
              double base, double rate, const std::vector<std::string>& names,
              std::size_t tree_count) {
  const char* name = objective == Objective::kBinaryLogistic ? "binary_logistic"
                                                             : "squared_error";
  if (format == DumpFormat::kText) {
    output << "objective=" << name << " num_features=" << names.size()
           << " tree_count=" << tree_count << '\n'
           << "base_score=" << base << " learning_rate=" << rate << '\n';
    return;
  }
  output << "{\"schema_version\":1,\"objective\":\"" << name
         << "\",\"num_features\":" << names.size()
         << ",\"tree_count\":" << tree_count << ",\"base_score\":" << base
         << ",\"learning_rate\":" << rate << ",\"feature_names\":[";
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (i) output << ',';
    internal::QuoteJson(output, names[i]);
  }
  output << ']';
}
void Write(std::ostream& output, const std::ostringstream& buffer) {
  output << buffer.str();
  if (!output) throw std::runtime_error("inspection write failed");
}
}  // namespace

void Gbdt::DumpModel(std::ostream& output, DumpFormat format, int tree_index,
                     const std::vector<std::string>& feature_names) const {
  if (trees_.empty()) throw std::logic_error("model is not fitted");
  ValidateFormat(format);
  if (tree_index < -1 || (tree_index >= 0 && static_cast<std::size_t>(
                                                 tree_index) >= trees_.size()))
    throw std::invalid_argument("tree index out of range");
  const auto names = FeatureNames(feature_names, num_features_);
  std::ostringstream buffer;
  Configure(buffer);
  Metadata(buffer, format, parameters_.objective, base_score_,
           parameters_.learning_rate, names, trees_.size());
  const bool json = format == DumpFormat::kJson;
  if (json) buffer << ",\"trees\":[";
  const std::size_t first = tree_index < 0 ? 0 : tree_index;
  const std::size_t last = tree_index < 0 ? trees_.size() : first + 1;
  for (std::size_t t = first; t < last; ++t) {
    const auto& tree = trees_[t];
    std::vector<int> depths(tree.size(), 0);
    if (json) {
      if (t != first) buffer << ',';
      buffer << "{\"tree_index\":" << t << ",\"nodes\":[";
    } else {
      buffer << "tree " << t << '\n';
    }
    for (std::size_t i = 0; i < tree.size(); ++i) {
      const auto& node = tree[i];
      if (json) {
        if (i) buffer << ',';
        buffer << "{\"node_id\":" << i;
      } else {
        buffer << std::string((depths[i] + 1) * 2, ' ') << "node " << i << ": ";
      }
      if (node.feature < 0) {
        const double contribution = parameters_.learning_rate * node.value;
        if (json)
          buffer << ",\"leaf_value\":" << node.value
                 << ",\"contribution\":" << contribution;
        else
          buffer << "leaf_value=" << node.value
                 << " contribution=" << contribution;
      } else {
        const int missing = node.missing_left ? node.left : node.right;
        if (json) {
          buffer << ",\"feature_index\":" << node.feature
                 << ",\"feature_name\":";
          internal::QuoteJson(buffer, names[node.feature]);
          buffer << ",\"threshold\":" << node.threshold
                 << ",\"left_id\":" << node.left
                 << ",\"right_id\":" << node.right
                 << ",\"missing_direction\":\""
                 << (node.missing_left ? "left" : "right") << '"';
        } else {
          internal::QuoteJson(buffer, names[node.feature]);
          buffer << " [f" << node.feature << "] <= " << node.threshold << " -> "
                 << node.left << "; > -> " << node.right << "; missing -> "
                 << missing;
          depths[node.left] = depths[node.right] = depths[i] + 1;
        }
      }
      buffer << (json ? "}" : "\n");
    }
    if (json) buffer << "]}";
  }
  if (json) buffer << "]}\n";
  Write(output, buffer);
}

void Gbdt::ExplainPrediction(
    MatrixView sample, std::ostream& output, DumpFormat format,
    const std::vector<std::string>& feature_names) const {
  ValidateFormat(format);
  const auto explanation = ExplainPrediction(sample);
  const auto names = FeatureNames(feature_names, num_features_);
  std::ostringstream buffer;
  Configure(buffer);
  const bool json = format == DumpFormat::kJson;
  Metadata(buffer, format, explanation.objective, explanation.base_score,
           explanation.learning_rate, names, explanation.trees.size());
  if (json)
    buffer << ",\"margin\":" << explanation.margin
           << ",\"prediction\":" << explanation.prediction << ",\"trees\":[";
  else
    buffer << "margin=" << explanation.margin
           << " prediction=" << explanation.prediction << '\n';
  for (const auto& tree : explanation.trees) {
    if (json) {
      if (tree.tree_index) buffer << ',';
      buffer << "{\"tree_index\":" << tree.tree_index << ",\"path\":[";
    } else {
      buffer << "tree " << tree.tree_index << '\n';
    }
    for (std::size_t i = 0; i < tree.path.size(); ++i) {
      const auto& step = tree.path[i];
      const bool missing = std::isnan(step.feature_value);
      const bool left =
          missing ? step.missing_left : step.feature_value <= step.threshold;
      const char* condition = missing ? "missing" : left ? "<=" : ">";
      if (json) {
        if (i) buffer << ',';
        buffer << "{\"node_id\":" << step.node_id
               << ",\"feature_index\":" << step.feature_index
               << ",\"feature_name\":";
        internal::QuoteJson(buffer, names[step.feature_index]);
        buffer << ",\"threshold\":" << step.threshold << ",\"feature_value\":";
        if (missing)
          buffer << "null";
        else
          buffer << step.feature_value;
        buffer << ",\"condition\":\"" << condition << "\",\"direction\":\""
               << (left ? "left" : "right") << "\",\"missing_direction\":\""
               << (step.missing_left ? "left" : "right")
               << "\",\"next_node_id\":" << step.next_node_id << '}';
      } else {
        buffer << "  node " << step.node_id << ": ";
        internal::QuoteJson(buffer, names[step.feature_index]);
        if (missing)
          buffer << " missing -> " << (step.missing_left ? "left" : "right");
        else
          buffer << " value=" << step.feature_value << ' ' << condition << ' '
                 << step.threshold;
        buffer << " -> node " << step.next_node_id << '\n';
      }
    }
    if (json)
      buffer << "],\"leaf_id\":" << tree.leaf_id
             << ",\"leaf_value\":" << tree.leaf_value
             << ",\"contribution\":" << tree.contribution << '}';
    else
      buffer << "  node " << tree.leaf_id << ": leaf_value=" << tree.leaf_value
             << " contribution=" << tree.contribution << '\n';
  }
  if (json) buffer << "]}\n";
  Write(output, buffer);
}
}  // namespace libtree
