// Native C++ comparison. XGBoost is loaded through its documented public C API.
#include <dlfcn.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "libtree/gbdt.h"

namespace {
using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
std::uint64_t PeakRssKib() {
  std::ifstream status("/proc/self/status");
  std::string key;
  while (status >> key) {
    if (key == "VmHWM:") {
      std::uint64_t kib;
      if (status >> kib) return kib;
      break;
    }
    status.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  throw std::runtime_error("Linux VmHWM unavailable");
}
struct Dataset {
  std::uint64_t train_rows, test_rows, cols;
  std::vector<float> train_x, train_y, test_x;
  explicit Dataset(const char* path) {
    std::ifstream input(path, std::ios::binary);
    auto read = [&](void* data, std::size_t bytes) {
      input.read(static_cast<char*>(data), bytes);
      if (!input) throw std::runtime_error("truncated benchmark data");
    };
    read(&train_rows, 8);
    read(&test_rows, 8);
    read(&cols, 8);
    if (!train_rows || !test_rows || !cols || cols > 10000 ||
        train_rows > 10000000 || test_rows > 10000000 ||
        train_rows * cols > 100000000 || test_rows * cols > 100000000) {
      throw std::runtime_error("invalid benchmark dimensions");
    }
    train_x.resize(train_rows * cols);
    train_y.resize(train_rows);
    test_x.resize(test_rows * cols);
    read(train_x.data(), train_x.size() * sizeof(float));
    read(train_y.data(), train_y.size() * sizeof(float));
    read(test_x.data(), test_x.size() * sizeof(float));
  }
};
// Function signatures from xgboost/include/xgboost/c_api.h. Stable exported
// ABI.
class Xgboost {
 public:
  using Size = std::uint64_t;
  using Handle = void*;
  explicit Xgboost(const char* path)
      : library_(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
    if (!library_) throw std::runtime_error(dlerror());
  }
  ~Xgboost() { dlclose(library_); }
  Xgboost(const Xgboost&) = delete;
  Xgboost& operator=(const Xgboost&) = delete;
  template <typename Function>
  Function Symbol(const char* name) {
    auto symbol = dlsym(library_, name);
    if (!symbol)
      throw std::runtime_error(std::string("missing symbol: ") + name);
    return reinterpret_cast<Function>(symbol);
  }
  void Check(int result) {
    if (result != 0)
      throw std::runtime_error(Symbol<const char* (*)()>("XGBGetLastError")());
  }

 private:
  void* library_;
};
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc < 5)
      throw std::runtime_error(
          "usage: native_benchmark data output binary engine [libxgboost.so]");
    Dataset data(argv[1]);
    const bool binary = std::string(argv[3]) == "1";
    std::vector<float> predictions;
    double fit_seconds, predict_seconds;
    if (std::string(argv[4]) == "libtree") {
      libtree::Parameters p;
      p.objective = binary ? libtree::Objective::kBinaryLogistic
                           : libtree::Objective::kSquaredError;
      libtree::Gbdt model(p);
      auto start = Clock::now();
      model.Fit({data.train_x.data(), data.train_rows, data.cols},
                data.train_y);
      fit_seconds = Seconds(start);
      start = Clock::now();
      predictions =
          model.Predict({data.test_x.data(), data.test_rows, data.cols});
      predict_seconds = Seconds(start);
    } else {
      if (argc != 6 || std::string(argv[4]) != "xgboost")
        throw std::runtime_error("invalid engine");
      Xgboost api(argv[5]);
      auto matrix =
          api.Symbol<int (*)(const float*, Xgboost::Size, Xgboost::Size, float,
                             Xgboost::Handle*, int)>(
              "XGDMatrixCreateFromMat_omp");
      auto set_info =
          api.Symbol<int (*)(Xgboost::Handle, const char*, const float*,
                             Xgboost::Size)>("XGDMatrixSetFloatInfo");
      auto create = api.Symbol<int (*)(const Xgboost::Handle*, Xgboost::Size,
                                       Xgboost::Handle*)>("XGBoosterCreate");
      auto param =
          api.Symbol<int (*)(Xgboost::Handle, const char*, const char*)>(
              "XGBoosterSetParam");
      auto update = api.Symbol<int (*)(Xgboost::Handle, int, Xgboost::Handle)>(
          "XGBoosterUpdateOneIter");
      auto predict =
          api.Symbol<int (*)(Xgboost::Handle, Xgboost::Handle, int, unsigned,
                             int, Xgboost::Size*, const float**)>(
              "XGBoosterPredict");
      auto free_matrix = api.Symbol<int (*)(Xgboost::Handle)>("XGDMatrixFree");
      auto free_booster = api.Symbol<int (*)(Xgboost::Handle)>("XGBoosterFree");
      Xgboost::Handle train = nullptr, test = nullptr, booster = nullptr;
      // RAII also covers failures after any successful allocation.
      auto cleanup = [&](int*) {
        if (booster) free_booster(booster);
        if (test) free_matrix(test);
        if (train) free_matrix(train);
      };
      int sentinel = 0;
      std::unique_ptr<int, decltype(cleanup)> guard(&sentinel, cleanup);
      auto start = Clock::now();
      api.Check(matrix(data.train_x.data(), data.train_rows, data.cols,
                       std::numeric_limits<float>::quiet_NaN(), &train, 1));
      api.Check(set_info(train, "label", data.train_y.data(), data.train_rows));
      api.Check(create(&train, 1, &booster));
      api.Check(param(booster, "objective",
                      binary ? "binary:logistic" : "reg:squarederror"));
      for (const auto& pair : std::vector<std::pair<const char*, const char*>>{
               {"tree_method", "hist"},
               {"max_depth", "4"},
               {"max_bin", "64"},
               {"eta", "0.1"},
               {"lambda", "1"},
               {"nthread", "1"},
               {"seed", "42"}}) {
        api.Check(param(booster, pair.first, pair.second));
      }
      for (int i = 0; i < 100; ++i) api.Check(update(booster, i, train));
      fit_seconds = Seconds(start);
      start = Clock::now();
      api.Check(matrix(data.test_x.data(), data.test_rows, data.cols,
                       std::numeric_limits<float>::quiet_NaN(), &test, 1));
      Xgboost::Size count;
      const float* output;
      api.Check(predict(booster, test, 0, 0, 0, &count, &output));
      predictions.assign(output, output + count);
      predict_seconds = Seconds(start);
    }
    std::ofstream output(argv[2], std::ios::binary);
    output.write(reinterpret_cast<const char*>(predictions.data()),
                 predictions.size() * sizeof(float));
    if (!output) throw std::runtime_error("cannot write predictions");
    std::cout << "{\"fit_seconds\":" << fit_seconds
              << ",\"predict_seconds\":" << predict_seconds
              << ",\"peak_rss_kib\":" << PeakRssKib() << "}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
