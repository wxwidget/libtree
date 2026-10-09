#include <cmath>
#include <iostream>
#include <vector>

#include "libtree/gbdt.h"

int main() {
  std::vector<float> x{0, 1, 2, 3, 4, 5};
  std::vector<float> y{0, 0, 0, 1, 1, 1};
  libtree::Parameters parameters;
  parameters.min_samples_leaf = 1;
  libtree::Gbdt model(parameters);
  model.Fit({x.data(), x.size(), 1}, y);
  const auto predictions = model.Predict({x.data(), x.size(), 1});
  for (std::size_t i = 0; i < y.size(); ++i) {
    if (std::abs(predictions[i] - y[i]) > 0.01) return 1;
    std::cout << predictions[i] << '\n';
  }
}
