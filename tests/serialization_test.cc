#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "libtree/c5.h"
#include "libtree/c_api.h"
#include "libtree/gbdt.h"

namespace {
void Check(bool value) {
  if (!value) throw std::runtime_error("serialization contract failed");
}
template <typename F>
void Throws(F function) {
  bool threw = false;
  try {
    function();
  } catch (const std::exception&) {
    threw = true;
  }
  Check(threw);
}
void C5Roundtrip() {
  float x[] = {0, 1, 2, 3, 4, 5, std::numeric_limits<float>::quiet_NaN()};
  libtree::MatrixView view{x, 7, 1};
  for (int depth : {0, 3, 4}) {
    libtree::C5Parameters p;
    p.max_depth = depth;
    p.min_samples_leaf = 1;
    p.num_threads = 2;
    if (depth == 4) p.pruning_alpha = 1;  // Retain unreachable pruned nodes.
    libtree::C5Classifier model(p), loaded;
    std::stringstream saved;
    Throws([&] { loaded.SaveModel(saved); });
    model.Fit(view, {0, 0, 1, 1, 2, 2, 0});
    model.SaveModel(saved);
    loaded.LoadModel(saved);
    Check(loaded.Predict(view) == model.Predict(view));
    Check(loaded.PredictProba(view) == model.PredictProba(view));
    Check(loaded.estimator_errors() == model.estimator_errors());
    std::ostringstream before, after;
    model.DumpRules(before, libtree::DumpFormat::kJson);
    loaded.DumpRules(after, libtree::DumpFormat::kJson);
    Check(before.str() == after.str());
    before.str("");
    after.str("");
    model.DumpModel(before, libtree::DumpFormat::kJson);
    loaded.DumpModel(after, libtree::DumpFormat::kJson);
    Check(before.str() == after.str());
    for (const std::string& bad :
         {std::string("bad"), saved.str().substr(0, 60),
          saved.str() + "extra"}) {
      std::istringstream input(bad);
      Throws([&] { loaded.LoadModel(input); });
      Check(loaded.Predict(view) == model.Predict(view));
    }
    std::ostringstream broken;
    broken.setstate(std::ios::badbit);
    Throws([&] { model.SaveModel(broken); });
  }
  std::vector<float> large(4000 * 20);
  std::vector<int> labels(4000);
  for (std::size_t r = 0; r < labels.size(); ++r) {
    large[r * 20] = static_cast<float>(r);
    labels[r] = r > 2000;
  }
  libtree::C5Parameters parameters;
  parameters.num_threads = 2;
  libtree::C5Classifier parallel(parameters), restored(parameters);
  libtree::MatrixView large_view{large.data(), 4000, 20};
  parallel.Fit(large_view, labels);
  std::stringstream snapshot;
  parallel.SaveModel(snapshot);
  restored.LoadModel(snapshot);
  Check(restored.PredictProba(large_view) == parallel.PredictProba(large_view));
  for (const char* invalid :
       {"LIBTREE_C5 2", "LIBTREE_C5 1\n0 1 8 1 1 0 0 0 1 2 1 0",
        "LIBTREE_C5 1\n1 1 8 1 1 0 0 0 0 2 1 0",
        "LIBTREE_C5 1\n1 1 8 1 1 0 0 0 1 2 1 0\n1 0 0",
        "LIBTREE_C5 1\n1 1 8 1 1 0 0 0 1 2 1 0\n1 0 1\n-1 -1 -1 0",
        "LIBTREE_C5 1\n1 1 8 1 1 0 0 0 1 2 1 0\n1 0 1\n-1 -1 -1 2 0 1 4 1",
        "LIBTREE_C5 1\n1 1 8 1 1 0 0 0 1 2 1 0\n1 0 1\n-1 0 -1 0 0 1 4 1",
        "LIBTREE_C5 1\n1 1 8 1 1 0 0 0 1 2 1 0\n1 0 1\n0 0 0 0 0 1 4 1"}) {
    std::istringstream input(invalid);
    Throws([&] { restored.LoadModel(input); });
    Check(restored.Predict(large_view) == parallel.Predict(large_view));
  }
}
void Abi() {
  float x[] = {0, 1, 2, 3};
  float y[] = {0, 0, 1, 1};
  int labels[] = {0, 0, 1, 1};
  for (bool c5 : {false, true}) {
    auto serialize = c5 ? LtC5Serialize : LtSerialize;
    auto deserialize = c5 ? LtC5Deserialize : LtDeserialize;
    auto free = c5 ? LtC5Free : LtFree;
    auto model = c5 ? LtC5Create(3, 2, 8, 1, 1, 0, 0, 0, 2)
                    : LtCreate(3, 2, 8, 1, .1, 1, 0, 0);
    size_t size = 0;
    Check(serialize(model, nullptr, 0, &size) == -1);
    Check((c5 ? LtC5Fit(model, x, 4, 1, labels, nullptr)
              : LtFit(model, x, 4, 1, y)) == 0);
    Check(serialize(model, nullptr, 0, &size) == 0 && size > 1);
    Check(serialize(model, nullptr, 0, nullptr) == -1);
    Check(serialize(nullptr, nullptr, 0, &size) == -1);
    char sentinel = 'X';
    Check(serialize(model, &sentinel, 1, &size) == -1 && sentinel == 'X');
    std::vector<char> buffer(size);
    Check(serialize(model, buffer.data(), buffer.size(), &size) == 0);
    Check(std::strlen(buffer.data()) + 1 == size);
    auto restored = deserialize(buffer.data(), size - 1, 2);
    Check(restored != nullptr);
    free(restored);
    Check(deserialize(nullptr, 1, 1) == nullptr);
    Check(deserialize(buffer.data(), 0, 1) == nullptr);
    Check(deserialize(buffer.data(), size / 2, 1) == nullptr);
    Check(deserialize(buffer.data(), size - 1, 0) == nullptr);
    free(model);
  }
}
}  // namespace
int main() {
  C5Roundtrip();
  Abi();
  std::cout
      << "Model snapshots preserve predictions, histories and explanations\n";
}
