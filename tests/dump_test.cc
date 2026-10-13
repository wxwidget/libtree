#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "libtree/gbdt.h"

namespace {
void Check(bool condition) {
  if (!condition) throw std::runtime_error("tree inspection check failed");
}
template <typename Function>
void Throws(Function function) {
  bool threw = false;
  try {
    function();
  } catch (const std::exception&) {
    threw = true;
  }
  Check(threw);
}
struct CommaDecimal : std::numpunct<char> {
  char do_decimal_point() const override { return ','; }
};
const char kModel[] =
    "LIBTREE_GBDT 1\n0 2 1 8 1 .5 0 0 1 1.25 2\n3\n"
    "0 1 2 .5 0 0\n-1 -1 -1 0 -2 1\n-1 -1 -1 0 2 1\n"
    "1\n-1 -1 -1 0 1 1\n";
void Run() {
  using libtree::DumpFormat;
  libtree::Gbdt model;
  std::ostringstream output;
  float value = 0;
  Throws([&] { model.DumpModel(output); });
  Throws([&] { model.ExplainPrediction({&value, 1, 1}); });
  std::istringstream input(kModel);
  model.LoadModel(input);
  for (float probe :
       {0.0f, 0.5f, 1.0f, std::numeric_limits<float>::quiet_NaN()}) {
    auto explanation = model.ExplainPrediction({&probe, 1, 1});
    Check(explanation.prediction == model.Predict({&probe, 1, 1})[0]);
    Check(explanation.base_score == 1.25 && explanation.learning_rate == .5);
    Check(explanation.trees.size() == 2);
    Check(explanation.trees[0].path.size() == 1);
    Check(explanation.trees[1].path.empty());
    Check(explanation.trees[0].path[0].feature_index == 0);
    Check(explanation.trees[0].path[0].missing_left == false);
    Check(explanation.trees[0].path[0].threshold == .5f);
    Check(explanation.margin == explanation.base_score +
                                    explanation.trees[0].contribution +
                                    explanation.trees[1].contribution);
    std::ostringstream text, json;
    model.ExplainPrediction({&probe, 1, 1}, text);
    model.ExplainPrediction({&probe, 1, 1}, json, DumpFormat::kJson, {"age"});
    Check(text.str().find("contribution") != std::string::npos);
    Check(json.str().find("\"feature_name\":\"age\"") != std::string::npos);
  }
  std::ostringstream json;
  json.imbue(std::locale(std::locale::classic(), new CommaDecimal));
  json.precision(3);
  model.DumpModel(json, DumpFormat::kJson);
  Check(json.precision() == 3);
  Check(json.str().find("\"threshold\":0.5") != std::string::npos);
  Check(json.str().find("\"missing_direction\":\"right\"") !=
        std::string::npos);
  Check(json.str().find("\"contribution\":-1") != std::string::npos);
  std::ostringstream selected;
  model.DumpModel(selected, DumpFormat::kText, 1, {"age"});
  Check(selected.str().find("tree 1") != std::string::npos);
  Check(selected.str().find("tree 0") == std::string::npos);
  const std::string controls = "age\"\\\b\f\n\r\t\1";
  std::ostringstream escaped;
  model.DumpModel(escaped, DumpFormat::kJson, 0, {controls});
  Check(escaped.str().find("age\\\"\\\\\\b\\f\\n\\r\\t\\u0001") !=
        std::string::npos);
  for (int index : {-2, 2})
    Throws([&] { model.DumpModel(output, DumpFormat::kText, index); });
  Throws([&] { model.DumpModel(output, static_cast<DumpFormat>(99)); });
  Throws([&] { model.DumpModel(output, DumpFormat::kJson, -1, {"a", "b"}); });
  Throws([&] {
    model.ExplainPrediction({&value, 1, 1}, output, DumpFormat::kJson,
                            {"a", "b"});
  });
  Throws([&] {
    model.ExplainPrediction({&value, 1, 1}, output,
                            static_cast<DumpFormat>(99));
  });
  Throws([&] { model.ExplainPrediction({&value, 0, 1}); });
  Throws([&] { model.ExplainPrediction({&value, 2, 1}); });
  Throws([&] { model.ExplainPrediction({nullptr, 1, 1}); });
  Throws([&] { model.ExplainPrediction({&value, 1, 2}); });
  std::ostringstream broken;
  broken.setstate(std::ios::badbit);
  Throws([&] { model.DumpModel(broken); });
  Throws([&] { model.ExplainPrediction({&value, 1, 1}, broken); });
  model.SetNumThreads(4);
  std::vector<std::future<std::string>> concurrent;
  for (int i = 0; i < 8; ++i)
    concurrent.push_back(std::async(std::launch::async, [&] {
      std::ostringstream exported;
      model.DumpModel(exported, DumpFormat::kJson);
      Check(model.ExplainPrediction({&value, 1, 1}).prediction ==
            model.Predict({&value, 1, 1})[0]);
      return exported.str();
    }));
  for (auto& exported : concurrent) Check(exported.get() == json.str());
  std::string binary(kModel);
  binary.replace(binary.find("\n0 2"), 4, "\n1 2");
  std::istringstream binary_input(binary);
  model.LoadModel(binary_input);
  auto explanation = model.ExplainPrediction({&value, 1, 1});
  Check(explanation.objective == libtree::Objective::kBinaryLogistic);
  Check(explanation.prediction == model.Predict({&value, 1, 1})[0]);
  model.DumpModel(output);
  Check(output.str().find("binary_logistic") != std::string::npos);
}
}  // namespace
int main() {
  try {
    Run();
    std::cout << "tree exports, paths, precision and errors passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
