#include "../src/parallel.h"

#include <future>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

int main() {
  try {
    for (int threads : {1, 2, 4}) {
      libtree::internal::ParallelExecutor executor(threads);
      std::vector<int> values(137);
      for (int repeat = 0; repeat < 20; ++repeat) {
        executor.For(values.size(), [&](int, std::size_t i) { values[i] = i; });
        if (std::accumulate(values.begin(), values.end(), 0) != 136 * 137 / 2)
          throw std::runtime_error("lost or duplicate tasks");
      }
      executor.For(0,
                   [](int, std::size_t) { throw std::runtime_error("empty"); });
      // Exceptions on the caller and on every possible worker must reach the
      // caller, and the executor must remain usable after every failure.
      for (std::size_t fail :
           {std::size_t{0}, std::size_t{100}, std::size_t{136}}) {
        bool caught = false;
        try {
          executor.For(137, [&](int, std::size_t i) {
            if (i == fail) throw std::runtime_error("expected task failure");
          });
        } catch (const std::runtime_error&) {
          caught = true;
        }
        if (!caught) throw std::runtime_error("exception not propagated");
        executor.For(1, [&](int, std::size_t) { values[0] = 42; });
        if (values[0] != 42) throw std::runtime_error("failed recovery");
      }
      std::vector<std::future<void>> callers;
      for (int call = 0; call < 4; ++call)
        callers.push_back(std::async(std::launch::async, [&] {
          std::vector<int> own(1000);
          executor.For(own.size(), [&](int, std::size_t i) { own[i] = i; });
          if (std::accumulate(own.begin(), own.end(), 0) != 999 * 1000 / 2)
            throw std::runtime_error("concurrent dispatch failed");
        }));
      for (auto& caller : callers) caller.get();
    }
    std::cout << "Parallel dispatch, exception recovery and concurrent callers "
                 "passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
