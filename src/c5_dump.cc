#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <locale>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "inspection.h"
#include "libtree/c5.h"

namespace libtree {
namespace {
std::string Quote(const std::string& s) {
  std::ostringstream out;
  out.exceptions(std::ios::badbit | std::ios::failbit);
  internal::QuoteJson(out, s);
  return out.str();
}
void Configure(std::ostringstream& out, DumpFormat format, std::size_t features,
               const std::vector<std::string>& names) {
  if (format != DumpFormat::kText && format != DumpFormat::kJson)
    throw std::invalid_argument("format must be text or json");
  if (!names.empty() && names.size() != features)
    throw std::invalid_argument("feature names must match columns");
  out.imbue(std::locale::classic());
  out.exceptions(std::ios::badbit | std::ios::failbit);
  out.precision(std::numeric_limits<double>::max_digits10);
}
std::string Name(int feature, const std::vector<std::string>& names) {
  return names.empty() ? "f" + std::to_string(feature) : names[feature];
}
void JsonRule(std::ostream& out, const ClassificationRule& rule,
              const std::vector<std::string>& names) {
  out << "{\"rule_id\":" << rule.rule_id
      << ",\"estimator_index\":" << rule.estimator_index
      << ",\"class_index\":" << rule.class_index
      << ",\"support\":" << rule.support
      << ",\"weighted_purity\":" << rule.purity << ",\"vote\":" << rule.vote
      << ",\"conditions\":[";
  for (std::size_t i = 0; i < rule.conditions.size(); ++i) {
    if (i) out << ',';
    const auto& c = rule.conditions[i];
    out << "{\"feature_index\":" << c.feature_index
        << ",\"feature_name\":" << Quote(Name(c.feature_index, names))
        << ",\"lower\":";
    if (c.has_lower)
      out << c.lower;
    else
      out << "null";
    out << ",\"upper\":";
    if (c.has_upper)
      out << c.upper;
    else
      out << "null";
    out << ",\"accepts_missing\":" << (c.accepts_missing ? "true" : "false")
        << '}';
  }
  out << "]}";
}
void TextRule(std::ostream& out, const ClassificationRule& rule,
              const std::vector<std::string>& names) {
  out << std::setprecision(std::numeric_limits<float>::max_digits10) << "R"
      << rule.rule_id << " [round " << rule.estimator_index << "]: IF ";
  if (rule.conditions.empty()) out << "TRUE";
  for (std::size_t i = 0; i < rule.conditions.size(); ++i) {
    if (i) out << " AND ";
    const auto& c = rule.conditions[i];
    const auto name = Quote(Name(c.feature_index, names));
    out << '(';
    if (c.has_lower) out << c.lower << " < ";
    out << name;
    if (c.has_upper) out << " <= " << c.upper;
    if (c.accepts_missing) out << " OR " << name << " IS MISSING";
    out << ')';
  }
  out << " THEN class " << rule.class_index << " [support=" << rule.support
      << std::setprecision(4) << ", weighted purity=" << rule.purity
      << ", vote=" << rule.vote << "]\n";
}
void Array(std::ostream& out, const std::vector<double>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << values[i];
  }
  out << ']';
}
void JsonPath(std::ostream& out, const ClassificationPath& path,
              const std::vector<std::string>& names) {
  out << "{\"estimator_index\":" << path.estimator_index
      << ",\"leaf_id\":" << path.leaf_id << ",\"rule_id\":" << path.rule_id
      << ",\"class_index\":" << path.class_index << ",\"vote\":" << path.vote
      << ",\"steps\":[";
  for (std::size_t i = 0; i < path.steps.size(); ++i) {
    if (i) out << ',';
    const auto& step = path.steps[i];
    out << "{\"node_id\":" << step.node_id
        << ",\"feature_index\":" << step.feature_index
        << ",\"feature_name\":" << Quote(Name(step.feature_index, names))
        << ",\"value\":";
    if (std::isnan(step.value))
      out << "null";
    else
      out << step.value;
    out << ",\"threshold\":" << step.threshold << ",\"missing_direction\":\""
        << (step.missing_left ? "left" : "right") << "\",\"direction\":\""
        << (step.took_left ? "left" : "right")
        << "\",\"next_node_id\":" << step.next_node_id << '}';
  }
  out << "]}";
}
void TextPath(std::ostream& out, const ClassificationPath& path,
              const std::vector<std::string>& names) {
  out << std::setprecision(std::numeric_limits<float>::max_digits10);
  out << "Path for tree " << path.estimator_index << ":\n";
  for (const auto& step : path.steps) {
    out << "  node " << step.node_id << ": "
        << Quote(Name(step.feature_index, names));
    if (std::isnan(step.value))
      out << " IS MISSING";
    else
      out << '=' << step.value << (step.took_left ? " <= " : " > ")
          << step.threshold;
    out << " -> " << (step.took_left ? "left" : "right") << " (node "
        << step.next_node_id << ")\n";
  }
  out << "  leaf " << path.leaf_id << ": THEN class " << path.class_index
      << std::setprecision(4) << " [vote=" << path.vote << ", rule=R"
      << path.rule_id << "]\n";
}
void Write(std::ostream& output, const std::string& text) {
  output << text;
  if (!output) throw std::runtime_error("rule output write failed");
}
}  // namespace
void C5Classifier::DumpModel(std::ostream& output, DumpFormat format,
                             int tree_index,
                             const std::vector<std::string>& names) const {
  RequireFitted();
  if (tree_index < -1 ||
      (tree_index >= 0 &&
       static_cast<std::size_t>(tree_index) >= learners_.size()))
    throw std::invalid_argument("tree index outside fitted estimators");
  std::ostringstream out;
  Configure(out, format, num_features_, names);
  const std::size_t begin =
      tree_index < 0 ? 0 : static_cast<std::size_t>(tree_index);
  const std::size_t end = tree_index < 0 ? learners_.size() : begin + 1;
  if (format == DumpFormat::kJson) {
    out << "{\"schema_version\":1,\"algorithm\":\"c5_style_samme\",\"classes\":"
        << num_classes_ << ",\"estimators\":" << learners_.size()
        << ",\"feature_names\":[";
    for (std::size_t f = 0; f < num_features_; ++f) {
      if (f) out << ',';
      out << Quote(Name(static_cast<int>(f), names));
    }
    out << "],\"trees\":[";
    for (std::size_t i = begin; i < end; ++i) {
      if (i != begin) out << ',';
      const auto& tree = learners_[i];
      out << "{\"tree_index\":" << i << ",\"vote\":" << tree.vote
          << ",\"weighted_error\":" << errors_[i] << ",\"nodes\":[";
      for (std::size_t id = 0; id < tree.nodes.size(); ++id) {
        if (id) out << ',';
        const auto& node = tree.nodes[id];
        out << "{\"node_id\":" << id
            << ",\"is_leaf\":" << (node.feature < 0 ? "true" : "false")
            << ",\"class_index\":" << node.label
            << ",\"support\":" << node.support
            << ",\"weighted_purity\":" << node.purity;
        if (node.feature < 0) {
          out << ",\"rule_id\":" << node.rule_id;
        } else {
          out << ",\"feature_index\":" << node.feature
              << ",\"feature_name\":" << Quote(Name(node.feature, names))
              << ",\"threshold\":" << node.threshold
              << ",\"missing_direction\":\""
              << (node.missing_left ? "left" : "right")
              << "\",\"left_child\":" << node.left
              << ",\"right_child\":" << node.right;
        }
        out << '}';
      }
      out << "]}";
    }
    out << "]}\n";
  } else {
    out << "C5-style SAMME trees: " << learners_.size() << " estimators, "
        << num_classes_ << " classes.\n"
        << "Finite values <= threshold go left; NaN follows the learned "
           "missing direction.\n";
    for (std::size_t i = begin; i < end; ++i) {
      const auto& tree = learners_[i];
      out << std::setprecision(4) << "Tree " << i << " [vote=" << tree.vote
          << ", weighted error=" << errors_[i] << "]\n";
      std::function<void(int, int, const char*)> walk =
          [&](int id, int depth, const char* branch) {
            const auto& node = tree.nodes[id];
            out << std::setprecision(std::numeric_limits<float>::max_digits10);
            out << std::string(static_cast<std::size_t>(depth) * 2, ' ')
                << branch << " node " << id << ": ";
            if (node.feature < 0) {
              out << "THEN class " << node.label << " [rule=R" << node.rule_id;
            } else {
              out << "IF " << Quote(Name(node.feature, names))
                  << " <= " << node.threshold << " [missing -> "
                  << (node.missing_left ? "left" : "right");
            }
            out << ", support=" << node.support << std::setprecision(4)
                << ", weighted purity=" << node.purity << "]\n";
            if (node.feature >= 0) {
              walk(node.left, depth + 1, "left");
              walk(node.right, depth + 1, "right");
            }
          };
      walk(0, 0, "root");
    }
  }
  Write(output, out.str());
}
void C5Classifier::DumpRules(std::ostream& output, DumpFormat format,
                             const std::vector<std::string>& names) const {
  RequireFitted();
  std::ostringstream out;
  Configure(out, format, num_features_, names);
  if (format == DumpFormat::kJson) {
    out << "{\"schema_version\":1,\"algorithm\":\"c5_style_samme\",\"classes\":"
        << num_classes_ << ",\"estimators\":" << learners_.size()
        << ",\"rules\":[";
    for (std::size_t i = 0; i < rules_.size(); ++i) {
      if (i) out << ',';
      JsonRule(out, rules_[i], names);
    }
    out << "]}\n";
  } else {
    out << "C5-style SAMME: " << learners_.size() << " rounds, " << num_classes_
        << " classes, " << rules_.size() << " rules.\n"
        << "One rule matches per round. Sum votes by class; ties choose the "
           "lowest class index.\n";
    for (const auto& rule : rules_) TextRule(out, rule, names);
  }
  Write(output, out.str());
}
void C5Classifier::Explain(MatrixView sample, std::ostream& output,
                           DumpFormat format,
                           const std::vector<std::string>& names) const {
  const auto explanation = Explain(sample);
  std::set<int> used;
  for (const auto& rule : explanation.matched_rules)
    for (const auto& condition : rule.conditions)
      used.insert(condition.feature_index);
  std::ostringstream out;
  Configure(out, format, num_features_, names);
  if (format == DumpFormat::kJson) {
    out << "{\"schema_version\":1,\"prediction\":" << explanation.prediction
        << ",\"votes\":";
    Array(out, explanation.votes);
    out << ",\"probabilities\":";
    Array(out, explanation.probabilities);
    out << ",\"matched_rules\":[";
    for (std::size_t i = 0; i < explanation.matched_rules.size(); ++i) {
      if (i) out << ',';
      JsonRule(out, explanation.matched_rules[i], names);
    }
    out << "],\"paths\":[";
    for (std::size_t i = 0; i < explanation.paths.size(); ++i) {
      if (i) out << ',';
      JsonPath(out, explanation.paths[i], names);
    }
    out << "],\"observations\":[";
    bool comma = false;
    for (int f : used) {
      if (comma) out << ',';
      comma = true;
      out << "{\"feature_index\":" << f
          << ",\"feature_name\":" << Quote(Name(f, names)) << ",\"value\":";
      if (std::isnan(sample.data[f]))
        out << "null";
      else
        out << sample.data[f];
      out << '}';
    }
    out << "]}\n";
  } else {
    out << "Observed features:";
    for (int f : used) {
      out << ' ' << Quote(Name(f, names)) << '=';
      if (std::isnan(sample.data[f]))
        out << "MISSING";
      else
        out << sample.data[f];
    }
    out << '\n';
    for (const auto& path : explanation.paths) TextPath(out, path, names);
    for (const auto& rule : explanation.matched_rules)
      TextRule(out, rule, names);
    out << "Total class votes: ";
    out << std::setprecision(4);
    Array(out, explanation.votes);
    out << "\nPredicted class: " << explanation.prediction
        << "\nUncalibrated probabilities: ";
    Array(out, explanation.probabilities);
    out << '\n';
  }
  Write(output, out.str());
}
}  // namespace libtree
