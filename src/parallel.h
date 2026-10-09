#ifndef LIBTREE_PARALLEL_H_
#define LIBTREE_PARALLEL_H_

#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace libtree {
namespace internal {
// Reusable fixed-size executor. The caller is worker zero. Static contiguous
// ranges keep each task's accumulation order independent of thread count.
// Calls from different prediction clients serialize at dispatch; workers never
// recursively dispatch. Exceptions propagate after every worker reaches the
// bar.
class ParallelExecutor {
 public:
  explicit ParallelExecutor(int threads) : threads_(threads) {
    try {
      for (int id = 1; id < threads; ++id)
        workers_.emplace_back([this, id] { Worker(id); });
    } catch (...) {
      Stop();
      throw;
    }
  }
  ~ParallelExecutor() { Stop(); }
  ParallelExecutor(const ParallelExecutor&) = delete;
  ParallelExecutor& operator=(const ParallelExecutor&) = delete;
  template <typename Function>
  void For(std::size_t count, Function function) {
    std::lock_guard<std::mutex> dispatch(dispatch_mutex_);
    struct Job {
      std::size_t count;
      int threads;
      Function function;
    } job{count, threads_, std::move(function)};
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // Non-owning stack job stays alive until the barrier. Dispatch itself
      // never allocates, including concurrent prediction calls.
      context_ = &job;
      work_ = [](void* context, int id) {
        auto& task = *static_cast<Job*>(context);
        const std::size_t begin = task.count * id / task.threads;
        const std::size_t end = task.count * (id + 1) / task.threads;
        for (std::size_t i = begin; i < end; ++i) task.function(id, i);
      };
      error_ = nullptr;
      remaining_ = workers_.size();
      ++generation_;
    }
    ready_.notify_all();
    Run(0);
    std::unique_lock<std::mutex> lock(mutex_);
    finished_.wait(lock, [&] { return remaining_ == 0; });
    work_ = nullptr;
    context_ = nullptr;
    if (error_) std::rethrow_exception(error_);
  }

 private:
  void Run(int id) {
    try {
      work_(context_, id);
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!error_) error_ = std::current_exception();
    }
  }
  void Worker(int id) {
    std::size_t seen = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      ready_.wait(lock, [&] { return stopping_ || generation_ != seen; });
      if (stopping_) return;
      seen = generation_;
      lock.unlock();
      Run(id);
      lock.lock();
      if (--remaining_ == 0) finished_.notify_one();
    }
  }
  void Stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    for (auto& worker : workers_) worker.join();
  }
  int threads_;
  std::vector<std::thread> workers_;
  std::mutex dispatch_mutex_, mutex_;
  std::condition_variable ready_, finished_;
  void (*work_)(void*, int) = nullptr;
  void* context_ = nullptr;
  std::exception_ptr error_;
  std::size_t generation_ = 0, remaining_ = 0;
  bool stopping_ = false;
};
}  // namespace internal
}  // namespace libtree
#endif  // LIBTREE_PARALLEL_H_
