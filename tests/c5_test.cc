#include "libtree/c5.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <locale>
#include <new>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "libtree/c_api.h"

// Deterministically fail one allocation, then restore normal allocation. This
// checks the strong refit guarantee without relying on actual system
// exhaustion.
thread_local int allocation_budget = -1;
void* operator new(std::size_t size) {
  if (allocation_budget == 0) {
    allocation_budget = -1;
    throw std::bad_alloc();
  }
  if (allocation_budget > 0) --allocation_budget;
  if (void* p = std::malloc(size ? size : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { ::operator delete(p); }

namespace {
void Check(bool value) {
  if (!value) throw std::runtime_error("C5 check failed");
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
bool Matches(const libtree::ClassificationRule& rule, const float* row) {
  for (const auto& c : rule.conditions) {
    const float v = row[c.feature_index];
    if (std::isnan(v)) {
      if (!c.accepts_missing) return false;
    } else if ((c.has_lower && v <= c.lower) || (c.has_upper && v > c.upper)) {
      return false;
    }
  }
  return true;
}
void Errors() {
  using libtree::C5Classifier;
  using libtree::C5Parameters;
  for (int test = 0; test < 15; ++test) {
    C5Parameters p;
    switch (test) {
      case 0:
        p.num_estimators = 0;
        break;
      case 1:
        p.num_estimators = 1001;
        break;
      case 2:
        p.max_depth = -1;
        break;
      case 3:
        p.max_depth = 33;
        break;
      case 4:
        p.max_bins = 1;
        break;
      case 5:
        p.max_bins = 257;
        break;
      case 6:
        p.min_samples_leaf = 0;
        break;
      case 7:
        p.num_threads = 0;
        break;
      case 8:
        p.num_threads = 257;
        break;
      case 9:
        p.learning_rate = 0;
        break;
      case 10:
        p.learning_rate = 2;
        break;
      case 11:
        p.learning_rate = std::numeric_limits<double>::quiet_NaN();
        break;
      case 12:
        p.pruning_alpha = -1;
        break;
      case 13:
        p.min_gain = std::numeric_limits<double>::infinity();
        break;
      case 14:
        p.criterion = static_cast<libtree::SplitCriterion>(8);
        break;
    }
    Throws([&] { C5Classifier invalid(p); });
  }
  C5Classifier m;
  float x[] = {0, 1, 2, 3};
  const libtree::MatrixView view{x, 4, 1};
  std::ostringstream out;
  Throws([&] { m.Predict(view); });
  Throws([&] { m.rules(); });
  Throws([&] { m.estimator_errors(); });
  Throws([&] { m.DumpRules(out); });
  for (auto bad : {libtree::MatrixView{x, 4, 0},
                   {nullptr, 4, 1},
                   {x, 100000001, 1},
                   {x, 4, 100001},
                   {x, std::numeric_limits<std::size_t>::max(), 2}})
    Throws([&] { m.Fit(bad, {0, 0, 1, 1}); });
  Throws([&] { m.Fit({x, 0, 1}, {}); });
  Throws([&] { m.Fit(view, {0}); });
  for (auto y : {std::vector<int>{-1, 0, 1, 1}, {256, 0, 1, 1}, {0, 0, 2, 2}})
    Throws([&] { m.Fit(view, y); });
  for (auto w : {std::vector<double>{1},
                 {0, 0, 0, 0},
                 {-1, 1, 1, 1},
                 {1, 1, 1, std::numeric_limits<double>::infinity()}})
    Throws([&] { m.Fit(view, {0, 0, 1, 1}, w); });
  m.Fit(view, {0, 0, 1, 1}, {0, 1e308, 1e308, 0});
  const auto before = m.Predict(view);
  float inf = std::numeric_limits<float>::infinity();
  Throws([&] { m.Fit({&inf, 1, 1}, {0}); });
  Check(m.Predict(view) == before);
  Throws([&] { m.Predict({x, 2, 2}); });
  Throws([&] { m.PredictProbaInto(view, nullptr); });
  Throws([&] { m.Explain({x, 0, 1}); });
  Throws([&] { m.Explain(view); });
  Throws([&] { m.DumpRules(out, static_cast<libtree::DumpFormat>(4)); });
  Throws([&] { m.DumpRules(out, libtree::DumpFormat::kJson, {"a", "b"}); });
  std::ostringstream broken;
  broken.setstate(std::ios::badbit);
  Throws([&] { m.DumpRules(broken); });
  Throws([&] { m.Explain({x, 1, 1}, broken); });
  m.PredictProbaInto({nullptr, 0, 1}, nullptr);
  Check(m.Predict({nullptr, 0, 1}).empty());
  Check(m.PredictProba({nullptr, 0, 1}).empty());
}
void Learning() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (auto criterion :
       {libtree::SplitCriterion::kGainRatio, libtree::SplitCriterion::kEntropy,
        libtree::SplitCriterion::kGini}) {
    libtree::C5Parameters p;
    p.min_samples_leaf = 1;
    p.pruning_alpha = 0;
    p.criterion = criterion;
    libtree::C5Classifier m(p);
    float values[] = {3, 3, 3, nan, nan, nan};
    m.Fit({values, 6, 1}, {0, 0, 0, 1, 1, 1});
    Check(m.Predict({values, 6, 1}) == std::vector<int>({0, 0, 0, 1, 1, 1}));
    Check(m.rules().size() == 2 && m.estimator_errors().size() == 1);
    m.Fit({values, 6, 1}, {0, 0, 0, 0, 0, 0});
    Check(m.num_classes() == 1 &&
          m.PredictProba({values, 6, 1}) == std::vector<double>(6, 1));
    std::ostringstream text, json;
    m.DumpRules(text);
    m.DumpRules(json, libtree::DumpFormat::kJson);
    Check(text.str().find("IF TRUE") != std::string::npos);
    Check(m.Explain({values, 1, 1}).prediction == 0);
    std::fill(std::begin(values), std::end(values), nan);
    m.Fit({values, 6, 1}, {0, 0, 0, 1, 1, 1});
    Check(m.rules().size() == 1);  // No learner better than random.
  }
  libtree::C5Parameters p;
  p.max_depth = 0;
  p.min_samples_leaf = 1;
  libtree::C5Classifier stump(p);
  float values[] = {0, 1, 2, 3};
  stump.Fit({values, 4, 1}, {0, 0, 0, 1});
  Check(stump.estimator_errors().size() == 1);  // Later weak round discarded.
}
void ParallelAndRules() {
  constexpr std::size_t rows = 5000, cols = 20;
  std::vector<float> x(rows * cols);
  std::vector<int> y(rows);
  std::mt19937 random(7);
  std::normal_distribution<float> normal;
  for (auto& v : x) v = normal(random);
  for (std::size_t r = 0; r < rows; ++r) {
    y[r] = x[r * cols] < -.5 ? 0 : (x[r * cols + 1] > .3 ? 1 : 2);
    if (r % 9 == 0) x[r * cols] = std::numeric_limits<float>::quiet_NaN();
  }
  std::string reference;
  std::vector<double> probability;
  for (int threads : {1, 4}) {
    libtree::C5Parameters p;
    p.num_threads = threads;
    p.num_estimators = 4;
    p.max_depth = 3;
    p.min_samples_leaf = 2;
    p.pruning_alpha = 0;
    libtree::C5Classifier m(p);
    m.Fit({x.data(), rows, cols}, y);
    const auto prediction = m.Predict({x.data(), rows, cols});
    std::ostringstream json;
    m.DumpRules(json, libtree::DumpFormat::kJson);
    if (threads == 1) {
      reference = json.str();
      probability = m.PredictProba({x.data(), rows, cols});
    } else {
      Check(json.str() == reference);
      Check(m.PredictProba({x.data(), rows, cols}) == probability);
    }
    std::size_t correct = 0;
    for (std::size_t r = 0; r < rows; ++r) {
      std::vector<double> votes(m.num_classes());
      std::size_t matched = 0;
      for (const auto& rule : m.rules())
        if (Matches(rule, x.data() + r * cols)) {
          votes[rule.class_index] += rule.vote;
          ++matched;
        }
      Check(matched == m.estimator_errors().size());
      Check(prediction[r] ==
            std::max_element(votes.begin(), votes.end()) - votes.begin());
      if (prediction[r] == y[r]) ++correct;
    }
    Check(correct > rows * 8 / 10);
    std::vector<float> large;
    for (int i = 0; i < 4; ++i) large.insert(large.end(), x.begin(), x.end());
    auto expected = m.Predict({large.data(), rows * 4, cols});
    std::vector<std::future<void>> clients;
    for (int i = 0; i < 4; ++i)
      clients.push_back(std::async(std::launch::async, [&] {
        Check(m.Predict({large.data(), rows * 4, cols}) == expected);
        const auto proba = m.PredictProba({large.data(), rows * 4, cols});
        Check(
            std::equal(probability.begin(), probability.end(), proba.begin()));
        std::ostringstream text;
        m.Explain({x.data(), 1, cols}, text);
      }));
    for (auto& client : clients) client.get();
  }
}
void Abi() {
  Check(LtC5Create(0, 2, 8, 1, 1, 0, 0, 0, 1) == nullptr);
  Check(LtC5ClassCount(nullptr) == 0 && LtC5ErrorCount(nullptr) == 0);
  auto handle = LtC5Create(4, 2, 8, 1, 1, .001, 0, 0, 1);
  Check(handle != nullptr);
  Check(LtC5ErrorCount(handle) == 0);
  float x[] = {0, 0, 1, 1};
  int y[] = {0, 0, 1, 1};
  Check(LtC5Fit(handle, x, 4, 1, nullptr, nullptr) == -1);
  Check(LtC5Fit(handle, x, 0, 1, y, nullptr) == -1);
  Check(LtC5Fit(nullptr, x, 4, 1, y, nullptr) == -1);
  Check(LtC5Fit(handle, x, 4, 1, y, nullptr) == 0);
  Check(LtC5ClassCount(handle) == 2 && LtC5ErrorCount(handle) == 1);
  double proba[8], error;
  Check(LtC5PredictProba(handle, x, 4, 1, proba) == 0);
  Check(LtC5PredictProba(handle, x, 4, 1, nullptr) == -1);
  Check(LtC5Errors(handle, &error, 1) == 0 && error == 0);
  Check(LtC5Errors(handle, nullptr, 1) == -1);
  Check(LtC5Errors(handle, &error, 2) == -1);
  size_t needed;
  Check(LtC5DumpRules(handle, 1, nullptr, 0, nullptr, 0, &needed) == 0);
  std::vector<char> output(needed);
  Check(LtC5DumpRules(handle, 1, nullptr, 0, output.data(), output.size(),
                      &needed) == 0);
  const char* name[] = {"age\"\\\n\1"};
  for (bool explain : {false, true}) {
    auto inspection = [&](const char* const* names, size_t count, char* buffer,
                          size_t capacity, size_t* size) {
      return explain ? LtC5Explain(handle, x, 1, 1, names, count, buffer,
                                   capacity, size)
                     : LtC5DumpRules(handle, 1, names, count, buffer, capacity,
                                     size);
    };
    Check(inspection(nullptr, 1, nullptr, 0, &needed) == -1);
    Check(inspection(nullptr, 0, nullptr, 1, &needed) == -1);
    Check(inspection(nullptr, 0, nullptr, 0, nullptr) == -1);
    const char* bad_name[] = {nullptr};
    Check(inspection(bad_name, 1, nullptr, 0, &needed) == -1);
    char sentinel = 'X';
    Check(inspection(nullptr, 0, &sentinel, 1, &needed) == -1 &&
          sentinel == 'X');
  }
  Check(LtC5Explain(handle, nullptr, 1, 1, nullptr, 0, nullptr, 0, &needed) ==
        -1);
  Check(LtC5Explain(handle, x, 1, 1, name, 1, nullptr, 0, &needed) == 0);
  output.resize(needed);
  Check(LtC5Explain(handle, x, 1, 1, name, 1, output.data(), output.size(),
                    &needed) == 0);
  Check(std::string(output.data()).find("\\n") != std::string::npos);
  LtC5Free(handle);
  LtC5Free(nullptr);
}
void AllocationFailures() {
  libtree::C5Parameters p;
  p.min_samples_leaf = 1;
  p.num_estimators = 3;
  libtree::C5Classifier model(p);
  float x[] = {0, 1, 2, 3, 4, 5};
  std::vector<int> y{0, 0, 1, 1, 2, 2};
  libtree::MatrixView view{x, 6, 1};
  model.Fit(view, y);
  auto before = model.Predict(view);
  for (int operation = 0; operation < 5; ++operation) {
    bool completed = false;
    for (int budget = 0; budget < 4096 && !completed; ++budget) {
      try {
        std::ostringstream out;
        allocation_budget = budget;
        if (operation == 0) model.Fit(view, y);
        if (operation == 1) model.PredictProba(view);
        if (operation == 2) model.Explain({x, 1, 1});
        if (operation == 3) model.DumpRules(out, libtree::DumpFormat::kJson);
        if (operation == 4)
          model.Explain({x, 1, 1}, out, libtree::DumpFormat::kJson);
        allocation_budget = -1;
        completed = true;
      } catch (const std::exception&) {
        Check(allocation_budget ==
              -1);  // Failure was injected, not spontaneous.
        allocation_budget = -1;
        Check(model.Predict(view) == before);
      }
    }
    Check(completed);
  }
}
}  // namespace
int main() {
  try {
    Errors();
    Learning();
    ParallelAndRules();
    Abi();
    AllocationFailures();
    std::cout << "C5 rules, SAMME, pruning, multiclass and ownership passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
