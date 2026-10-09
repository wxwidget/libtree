// Native C++ comparison. XGBoost is loaded through its documented public C API.
#include <dlfcn.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "libtree/gbdt.h"

namespace {
using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
std::string Precise(double value) {
  std::ostringstream output;
  output << std::setprecision(17) << value;
  return output.str();
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
class NativeLibrary {
 public:
  using Size = std::uint64_t;
  using Handle = void*;
  explicit NativeLibrary(const char* path,
                         const char* error_symbol = "XGBGetLastError")
      : library_(dlopen(path, RTLD_NOW | RTLD_LOCAL)),
        error_symbol_(error_symbol) {
    if (!library_) throw std::runtime_error(dlerror());
  }
  ~NativeLibrary() { dlclose(library_); }
  NativeLibrary(const NativeLibrary&) = delete;
  NativeLibrary& operator=(const NativeLibrary&) = delete;
  template <typename Function>
  Function Symbol(const char* name) {
    auto symbol = dlsym(library_, name);
    if (!symbol)
      throw std::runtime_error(std::string("missing symbol: ") + name);
    return reinterpret_cast<Function>(symbol);
  }
  void Check(int result) {
    if (result != 0)
      throw std::runtime_error(Symbol<const char* (*)()>(error_symbol_)());
  }

 private:
  void* library_;
  const char* error_symbol_;
};
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc < 5)
      throw std::runtime_error(
          "usage: native_benchmark data output binary engine [libxgboost.so]");
    Dataset data(argv[1]);
    const bool binary = std::string(argv[3]) == "1";
    libtree::Parameters p;
    p.objective = binary ? libtree::Objective::kBinaryLogistic
                         : libtree::Objective::kSquaredError;
    if (argc == 13 || argc == 14) {
      p.num_trees = std::stoi(argv[6]);
      p.max_depth = std::stoi(argv[7]);
      p.max_bins = std::stoi(argv[8]);
      p.min_samples_leaf = std::stoi(argv[9]);
      p.learning_rate = std::stod(argv[10]);
      p.l2 = std::stod(argv[11]);
      p.min_gain = std::stod(argv[12]);
      if (argc == 14) p.num_threads = std::stoi(argv[13]);
    } else if (argc != 5 && argc != 6)
      throw std::runtime_error("invalid benchmark arguments");
    const libtree::Gbdt validated(p);
    std::vector<float> predictions;
    double fit_seconds, predict_seconds;
    if (std::string(argv[4]) == "libtree") {
      libtree::Gbdt model(p);
      auto start = Clock::now();
      model.Fit({data.train_x.data(), data.train_rows, data.cols},
                data.train_y);
      fit_seconds = Seconds(start);
      start = Clock::now();
      predictions =
          model.Predict({data.test_x.data(), data.test_rows, data.cols});
      predict_seconds = Seconds(start);
    } else if (std::string(argv[4]) == "lightgbm") {
      if (argc != 6 && argc != 13 && argc != 14)
        throw std::runtime_error("LightGBM library path required");
      NativeLibrary api(argv[5], "LGBM_GetLastError");
      using Handle = void*;
      auto matrix =
          api.Symbol<int (*)(const void*, int, std::int32_t, std::int32_t, int,
                             const char*, Handle, Handle*)>(
              "LGBM_DatasetCreateFromMat");
      auto labels =
          api.Symbol<int (*)(Handle, const char*, const void*, int, int)>(
              "LGBM_DatasetSetField");
      auto create = api.Symbol<int (*)(Handle, const char*, Handle*)>(
          "LGBM_BoosterCreate");
      auto update =
          api.Symbol<int (*)(Handle, int*)>("LGBM_BoosterUpdateOneIter");
      auto predict = api.Symbol<int (*)(Handle, const void*, int, std::int32_t,
                                        std::int32_t, int, int, int, int,
                                        const char*, std::int64_t*, double*)>(
          "LGBM_BoosterPredictForMat");
      auto free_dataset = api.Symbol<int (*)(Handle)>("LGBM_DatasetFree");
      auto free_booster = api.Symbol<int (*)(Handle)>("LGBM_BoosterFree");
      std::ostringstream options;
      options
          << std::setprecision(17)
          << "objective=" << (binary ? "binary" : "regression")
          << " max_depth=" << p.max_depth
          << " num_leaves=" << (1 << p.max_depth) << " max_bin=" << p.max_bins
          << " min_data_in_leaf=" << p.min_samples_leaf
          << " learning_rate=" << p.learning_rate << " lambda_l2=" << p.l2
          << " min_gain_to_split=" << p.min_gain
          << " num_threads=" << p.num_threads
          << " verbosity=-1 seed=42 data_random_seed=42 "
             "deterministic=true"
          << " force_col_wise=true feature_pre_filter=false min_data_in_bin=1";
      Handle train = nullptr, booster = nullptr;
      auto cleanup = [&](int*) {
        if (booster) free_booster(booster);
        if (train) free_dataset(train);
      };
      int sentinel = 0;
      std::unique_ptr<int, decltype(cleanup)> guard(&sentinel, cleanup);
      auto start = Clock::now();
      api.Check(matrix(data.train_x.data(), 0, data.train_rows, data.cols, 1,
                       options.str().c_str(), nullptr, &train));
      api.Check(
          labels(train, "label", data.train_y.data(), data.train_rows, 0));
      api.Check(create(train, options.str().c_str(), &booster));
      for (int i = 0; i < p.num_trees; ++i) {
        int finished = 0;
        api.Check(update(booster, &finished));
        if (finished) break;
      }
      fit_seconds = Seconds(start);
      start = Clock::now();
      std::vector<double> output(data.test_rows);
      std::int64_t count = 0;
      api.Check(predict(
          booster, data.test_x.data(), 0, data.test_rows, data.cols, 1, 0, 0,
          -1, ("num_threads=" + std::to_string(p.num_threads)).c_str(), &count,
          output.data()));
      if (count != static_cast<std::int64_t>(data.test_rows))
        throw std::runtime_error("LightGBM prediction size mismatch");
      predictions.assign(output.begin(), output.end());
      predict_seconds = Seconds(start);
    } else {
      if ((argc != 6 && argc != 13 && argc != 14) ||
          std::string(argv[4]) != "xgboost")
        throw std::runtime_error("invalid engine");
      NativeLibrary api(argv[5]);
      auto matrix = api.Symbol<int (*)(
          const float*, NativeLibrary::Size, NativeLibrary::Size, float,
          NativeLibrary::Handle*, int)>("XGDMatrixCreateFromMat_omp");
      auto set_info =
          api.Symbol<int (*)(NativeLibrary::Handle, const char*, const float*,
                             NativeLibrary::Size)>("XGDMatrixSetFloatInfo");
      auto create =
          api.Symbol<int (*)(const NativeLibrary::Handle*, NativeLibrary::Size,
                             NativeLibrary::Handle*)>("XGBoosterCreate");
      auto param =
          api.Symbol<int (*)(NativeLibrary::Handle, const char*, const char*)>(
              "XGBoosterSetParam");
      auto update =
          api.Symbol<int (*)(NativeLibrary::Handle, int,
                             NativeLibrary::Handle)>("XGBoosterUpdateOneIter");
      auto predict = api.Symbol<int (*)(
          NativeLibrary::Handle, NativeLibrary::Handle, int, unsigned, int,
          NativeLibrary::Size*, const float**)>("XGBoosterPredict");
      auto free_matrix =
          api.Symbol<int (*)(NativeLibrary::Handle)>("XGDMatrixFree");
      auto free_booster =
          api.Symbol<int (*)(NativeLibrary::Handle)>("XGBoosterFree");
      NativeLibrary::Handle train = nullptr, test = nullptr, booster = nullptr;
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
                       std::numeric_limits<float>::quiet_NaN(), &train,
                       p.num_threads));
      api.Check(set_info(train, "label", data.train_y.data(), data.train_rows));
      api.Check(create(&train, 1, &booster));
      api.Check(param(booster, "objective",
                      binary ? "binary:logistic" : "reg:squarederror"));
      for (const auto& pair : std::vector<std::pair<std::string, std::string>>{
               {"tree_method", "hist"},
               {"max_depth", std::to_string(p.max_depth)},
               {"max_bin", std::to_string(p.max_bins)},
               {"eta", Precise(p.learning_rate)},
               {"lambda", Precise(p.l2)},
               {"gamma", Precise(p.min_gain)},
               {"nthread", std::to_string(p.num_threads)},
               {"seed", "42"}}) {
        api.Check(param(booster, pair.first.c_str(), pair.second.c_str()));
      }
      for (int i = 0; i < p.num_trees; ++i)
        api.Check(update(booster, i, train));
      fit_seconds = Seconds(start);
      start = Clock::now();
      api.Check(matrix(data.test_x.data(), data.test_rows, data.cols,
                       std::numeric_limits<float>::quiet_NaN(), &test,
                       p.num_threads));
      NativeLibrary::Size count;
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
