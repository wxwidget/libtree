#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "libtree/gbdt.h"

namespace {
const char kHelp[] =
    "xgbt: LibTree C++17 GBDT (not the XGBoost executable)\n"
    "  xgbt train --train labeled.csv [--test features.csv] [--model model.lt] "
    "[--output predictions.csv]\n"
    "  xgbt predict --model model.lt --data features.csv --output "
    "predictions.csv\n"
    "CSV: numeric, optional --header; first training column is label. "
    "Empty/nan feature = missing.\n"
    "Parameters: --objective regression|binary --trees 100 --depth 4 --bins "
    "64\n"
    "  --min-leaf 5 --learning-rate 0.1 (alias --rate) --l2 1 --min-gain 0\n"
    "Long aliases: --num-trees, --max-depth, --max-bins, --min-samples-leaf\n"
    "stdout: JSON metrics; predictions: one value/probability per line.\n";
struct Options {
  libtree::Parameters parameters;
  std::map<std::string, std::string> paths;
  std::string command;
  bool header = false;
};
double Number(const std::string& text) {
  std::size_t end = 0;
  double value = std::stod(text, &end);
  if (end != text.size() || !std::isfinite(value))
    throw std::invalid_argument("invalid number: " + text);
  return value;
}
int Integer(const std::string& text) {
  std::size_t end = 0;
  int value = std::stoi(text, &end);
  if (end != text.size())
    throw std::invalid_argument("invalid integer: " + text);
  return value;
}
Options Parse(int argc, char** argv) {
  Options options;
  options.command = argv[1];
  if (options.command != "train" && options.command != "predict")
    throw std::invalid_argument("expected train or predict");
  std::map<std::string, bool> seen;
  for (int i = 2; i < argc; ++i) {
    std::string flag = argv[i];
    if (flag == "--num-trees") flag = "--trees";
    if (flag == "--max-depth") flag = "--depth";
    if (flag == "--max-bins") flag = "--bins";
    if (flag == "--min-samples-leaf") flag = "--min-leaf";
    if (flag == "--learning-rate") flag = "--rate";
    if (seen[flag]) throw std::invalid_argument("duplicate option: " + flag);
    seen[flag] = true;
    if (flag == "--header") {
      options.header = true;
      continue;
    }
    if (++i >= argc) throw std::invalid_argument("missing option value");
    const std::string value = argv[i];
    if (flag == "--train" || flag == "--test" || flag == "--model" ||
        flag == "--output" || flag == "--data") {
      options.paths[flag] = value;
      continue;
    }
    if (options.command == "predict")
      throw std::invalid_argument("predict parameters come from the model");
    auto& p = options.parameters;
    if (flag == "--trees")
      p.num_trees = Integer(value);
    else if (flag == "--depth")
      p.max_depth = Integer(value);
    else if (flag == "--bins")
      p.max_bins = Integer(value);
    else if (flag == "--min-leaf")
      p.min_samples_leaf = Integer(value);
    else if (flag == "--rate")
      p.learning_rate = Number(value);
    else if (flag == "--l2")
      p.l2 = Number(value);
    else if (flag == "--min-gain")
      p.min_gain = Number(value);
    else if (flag == "--objective") {
      if (value != "regression" && value != "binary")
        throw std::invalid_argument("invalid objective");
      p.objective = value == "binary" ? libtree::Objective::kBinaryLogistic
                                      : libtree::Objective::kSquaredError;
    } else
      throw std::invalid_argument("unknown option: " + flag);
  }
  const auto& paths = options.paths;
  if (options.command == "train") {
    if (!paths.count("--train") || paths.count("--data"))
      throw std::invalid_argument(
          "train requires --train and does not accept --data");
  } else if (!paths.count("--model") || !paths.count("--data") ||
             !paths.count("--output") || paths.count("--train") ||
             paths.count("--test")) {
    throw std::invalid_argument(
        "predict requires --model, --data and --output");
  }
  for (const auto& output : paths) {
    if (output.first != "--output" &&
        !(options.command == "train" && output.first == "--model"))
      continue;
    for (const auto& input : paths) {
      if (input.first == output.first) continue;
      if (std::filesystem::weakly_canonical(input.second) ==
          std::filesystem::weakly_canonical(output.second)) {
        throw std::invalid_argument("input and output paths must differ");
      }
    }
  }
  return options;
}
struct Csv {
  std::vector<float> x, y;
  std::size_t rows = 0, columns = 0;
  libtree::MatrixView view() const { return {x.data(), rows, columns}; }
};
Csv ReadCsv(const std::string& path, bool labeled, bool header) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open input: " + path);
  Csv result;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (header && line_number == 1) continue;
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    std::vector<float> values;
    std::size_t begin = 0;
    while (true) {
      const std::size_t comma = line.find(',', begin);
      std::string token = line.substr(
          begin, comma == std::string::npos ? comma : comma - begin);
      const auto first = token.find_first_not_of(" \t\r");
      if (first == std::string::npos)
        token.clear();
      else
        token =
            token.substr(first, token.find_last_not_of(" \t\r") - first + 1);
      float value = std::numeric_limits<float>::quiet_NaN();
      if (!token.empty()) {
        std::size_t consumed = 0;
        value = std::stof(token, &consumed);
        if (consumed != token.size() || std::isinf(value))
          throw std::invalid_argument("invalid CSV value at line " +
                                      std::to_string(line_number));
      }
      values.push_back(value);
      if (comma == std::string::npos) break;
      begin = comma + 1;
    }
    const std::size_t columns = values.size() - (labeled ? 1 : 0);
    if (columns == 0 || (result.rows && columns != result.columns))
      throw std::invalid_argument("CSV column count mismatch");
    result.columns = columns;
    if (labeled) result.y.push_back(values[0]);
    result.x.insert(result.x.end(), values.begin() + (labeled ? 1 : 0),
                    values.end());
    ++result.rows;
  }
  if (input.bad() || result.rows == 0)
    throw std::invalid_argument("empty or unreadable CSV");
  return result;
}
void WriteAtomic(const std::string& path,
                 const std::function<void(std::ostream&)>& write) {
  const std::string temporary =
      path + ".tmp-" +
      std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count());
  if (std::filesystem::exists(temporary))
    throw std::runtime_error("temporary file already exists");
  try {
    std::ofstream output(temporary);
    if (!output) throw std::runtime_error("cannot open output: " + path);
    write(output);
    output.close();
    if (!output) throw std::runtime_error("output write failed");
    std::filesystem::rename(temporary, path);
  } catch (...) {
    std::filesystem::remove(temporary);
    throw;
  }
}
}  // namespace
int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::cout << kHelp;
    return 0;
  }
  if (argc < 2) {
    std::cerr << kHelp;
    return 2;
  }
  try {
    const Options options = Parse(argc, argv);
    libtree::Gbdt model(options.parameters);
    Csv train, test;
    double fit_seconds = 0;
    if (options.command == "train") {
      train = ReadCsv(options.paths.at("--train"), true, options.header);
      if (options.paths.count("--test"))
        test = ReadCsv(options.paths.at("--test"), false, options.header);
      const auto start = std::chrono::steady_clock::now();
      model.Fit(train.view(), train.y);
      fit_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start)
                        .count();
    } else {
      std::ifstream input(options.paths.at("--model"));
      model.LoadModel(input);
      test = ReadCsv(options.paths.at("--data"), false, options.header);
    }
    const auto start = std::chrono::steady_clock::now();
    const auto prediction =
        model.Predict(test.rows ? test.view() : train.view());
    const double predict_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    if (options.command == "train" && options.paths.count("--model")) {
      WriteAtomic(options.paths.at("--model"),
                  [&](std::ostream& output) { model.SaveModel(output); });
    }
    if (options.paths.count("--output"))
      WriteAtomic(options.paths.at("--output"), [&](std::ostream& output) {
        output << std::setprecision(std::numeric_limits<float>::max_digits10);
        for (float value : prediction) output << value << '\n';
      });
    std::cout << "{\"trees\":" << model.num_trees()
              << ",\"rows\":" << prediction.size()
              << ",\"fit_seconds\":" << fit_seconds
              << ",\"predict_seconds\":" << predict_seconds;
    if (!model.training_loss().empty())
      std::cout << ",\"training_loss\":" << model.training_loss().back();
    std::cout << "}\n";
  } catch (const std::exception& error) {
    std::cerr << "xgbt: " << error.what() << '\n';
    return 1;
  }
}
