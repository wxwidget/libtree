// Benchmark harness; prepared inputs use the same layout as native_benchmark.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "libtree/c5.h"

int main(int argc, char** argv) {
  try {
    if (argc != 12 && argc != 13)
      throw std::invalid_argument(
          "usage: c5_benchmark data output rounds depth bins min_leaf rate "
          "alpha gain criterion threads");
    std::ifstream input(argv[1], std::ios::binary);
    auto read = [&](void* out, std::size_t bytes) {
      input.read(static_cast<char*>(out), bytes);
      if (!input) throw std::runtime_error("truncated benchmark input");
    };
    std::uint64_t rows, test_rows, cols;
    read(&rows, 8);
    read(&test_rows, 8);
    read(&cols, 8);
    if (!rows || !test_rows || !cols || cols > 10000 || rows > 10000000 ||
        test_rows > 10000000 || rows * cols > 100000000 ||
        test_rows * cols > 100000000)
      throw std::invalid_argument("invalid benchmark dimensions");
    std::vector<float> x(rows * cols), labels(rows), test(test_rows * cols);
    read(x.data(), x.size() * sizeof(float));
    read(labels.data(), labels.size() * sizeof(float));
    read(test.data(), test.size() * sizeof(float));
    std::vector<int> y;
    for (float label : labels) {
      if (!std::isfinite(label) || label < 0 || label > 255 ||
          label != std::floor(label))
        throw std::invalid_argument("invalid benchmark label");
      y.push_back(static_cast<int>(label));
    }
    libtree::C5Parameters p;
    p.num_estimators = std::stoi(argv[3]);
    p.max_depth = std::stoi(argv[4]);
    p.max_bins = std::stoi(argv[5]);
    p.min_samples_leaf = std::stoi(argv[6]);
    p.learning_rate = std::stod(argv[7]);
    p.pruning_alpha = std::stod(argv[8]);
    p.min_gain = std::stod(argv[9]);
    p.criterion = static_cast<libtree::SplitCriterion>(std::stoi(argv[10]));
    p.num_threads = std::stoi(argv[11]);
    libtree::C5Classifier model(p);
    const auto start = std::chrono::steady_clock::now();
    std::vector<double> weights;
    if (argc == 13) {
      const int balanced = std::stoi(argv[12]);
      if (balanced != 0 && balanced != 1)
        throw std::invalid_argument("invalid balanced flag");
      if (balanced) {
        const int classes = *std::max_element(y.begin(), y.end()) + 1;
        std::vector<std::size_t> counts(classes);
        for (int label : y) ++counts[label];
        for (int label : y)
          weights.push_back(static_cast<double>(rows) /
                            (classes * counts[label]));
      }
    }
    model.Fit({x.data(), rows, cols}, y, weights);
    const auto trained = std::chrono::steady_clock::now();
    const auto prediction = model.PredictProba({test.data(), test_rows, cols});
    const auto predicted = std::chrono::steady_clock::now();
    const auto classes = model.Predict({test.data(), test_rows, cols});
    const auto classified = std::chrono::steady_clock::now();
    std::ofstream output(argv[2], std::ios::binary);
    output.write(reinterpret_cast<const char*>(prediction.data()),
                 prediction.size() * sizeof(double));
    output.close();
    const auto train_prediction = model.PredictProba({x.data(), rows, cols});
    std::ofstream train_output(std::string(argv[2]) + ".train",
                               std::ios::binary);
    train_output.write(reinterpret_cast<const char*>(train_prediction.data()),
                       train_prediction.size() * sizeof(double));
    train_output.close();
    if (!output || !train_output)
      throw std::runtime_error("prediction output failed");
    std::ofstream rules_output(std::string(argv[2]) + ".rules.json");
    model.DumpRules(rules_output, libtree::DumpFormat::kJson);
    rules_output.close();
    if (!rules_output) throw std::runtime_error("rule output failed");
    for (std::size_t r = 0; r < test_rows; ++r) {
      const auto begin = prediction.begin() + r * model.num_classes();
      if (classes[r] !=
          std::max_element(begin, begin + model.num_classes()) - begin)
        throw std::runtime_error("class/probability prediction mismatch");
    }
    std::size_t conditions = 0;
    for (const auto& rule : model.rules()) conditions += rule.conditions.size();
    std::cout << std::setprecision(17) << "{\"fit_seconds\":"
              << std::chrono::duration<double>(trained - start).count()
              << ",\"predict_seconds\":"
              << std::chrono::duration<double>(predicted - trained).count()
              << ",\"predict_label_seconds\":"
              << std::chrono::duration<double>(classified - predicted).count()
              << ",\"rules\":" << model.rules().size()
              << ",\"conditions\":" << conditions
              << ",\"estimators\":" << model.estimator_errors().size() << "}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
