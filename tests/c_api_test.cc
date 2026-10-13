#include "libtree/c_api.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void Check(bool condition) {
  if (!condition) throw std::runtime_error("C API check failed");
}
}  // namespace
int main() {
  Check(LtCreate(0, 2, 64, 1, .1, 1, 0, 0) == nullptr);
  Check(std::strlen(LtLastError()) != 0);
  Check(LtCreate(1, 2, 64, 1, .1, 1, 0, 2) == nullptr);
  Check(LtLossCount(nullptr) == 0);
  Check(LtLoss(nullptr, nullptr, 0) == -1);
  Check(LtFit(nullptr, nullptr, 0, 0, nullptr) == -1);
  LtFree(nullptr);
  float x[]{0, 1, 2, 3};
  float y[]{0, 0, 1, 1};
  float out[4];
  size_t required = 0;
  Check(LtDumpModel(nullptr, 0, -1, nullptr, 0, nullptr, 0, &required) == -1);
  Check(LtExplainPrediction(nullptr, x, 1, 0, nullptr, 0, nullptr, 0,
                            &required) == -1);
  LtModel inspected = LtCreate(2, 1, 8, 1, .2, 0, 0, 0);
  Check(LtDumpModel(inspected, 0, -1, nullptr, 0, nullptr, 0, &required) == -1);
  Check(LtExplainPrediction(inspected, x, 1, 0, nullptr, 0, nullptr, 0,
                            &required) == -1);
  Check(LtFit(inspected, x, 4, 1, y) == 0);
  const char* names[]{"age"};
  const char* invalid_names[]{nullptr};
  for (bool explain : {false, true}) {
    auto inspect = [&](int format, const char* const* labels, size_t count,
                       char* buffer, size_t capacity, size_t* needed) {
      return explain ? LtExplainPrediction(inspected, x, 1, format, labels,
                                           count, buffer, capacity, needed)
                     : LtDumpModel(inspected, format, -1, labels, count, buffer,
                                   capacity, needed);
    };
    Check(inspect(0, nullptr, 0, nullptr, 0, nullptr) == -1);
    Check(inspect(0, nullptr, 1, nullptr, 0, &required) == -1);
    Check(inspect(0, invalid_names, 1, nullptr, 0, &required) == -1);
    Check(inspect(99, nullptr, 0, nullptr, 0, &required) == -1);
    for (int format : {0, 1}) {
      Check(inspect(format, names, 1, nullptr, 0, &required) == 0);
      Check(required > 1);
      char preserved[] = "kept";
      Check(inspect(format, names, 1, preserved, sizeof(preserved),
                    &required) == -1);
      Check(std::strcmp(preserved, "kept") == 0);
      Check(inspect(format, names, 1, nullptr, 1, &required) == -1);
      std::vector<char> buffer(required);
      Check(inspect(format, names, 1, buffer.data(), buffer.size(),
                    &required) == 0);
      Check(std::strlen(buffer.data()) + 1 == required);
      Check(std::strstr(buffer.data(), "age") != nullptr);
    }
  }
  Check(LtExplainPrediction(inspected, nullptr, 1, 0, nullptr, 0, nullptr, 0,
                            &required) == -1);
  LtFree(inspected);
  // Repeated ownership cycles exercise leak detection and ABI error cleanup.
  for (int iteration = 0; iteration < 100; ++iteration) {
    LtModel model = LtCreate(30, 2, 8, 1, .2, 0, 0, 0);
    Check(model != nullptr);
    Check(LtPredict(model, x, 4, 1, out) == -1);
    Check(LtFit(model, x, 4, 1, y) == 0);
    Check(LtPredict(model, x, 4, 1, out) == 0);
    Check(out[0] < .01 && out[3] > .99);
    Check(LtPredict(model, x, 4, 1, nullptr) == -1);
    Check(LtPredict(model, nullptr, 0, 1, nullptr) == 0);
    Check(LtLossCount(model) == 30);
    std::vector<double> loss(30);
    Check(LtLoss(model, loss.data(), 30) == 0);
    Check(LtLoss(model, nullptr, 30) == -1);
    Check(LtLoss(model, loss.data(), 29) == -1);
    Check(loss.back() < loss.front());
    LtFree(model);
  }
  std::cout << "100 ownership cycles and C ABI checks passed\n";
}
