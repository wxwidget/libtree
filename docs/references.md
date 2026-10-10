# Research references and implementation map

[中文](references.zh-CN.md)

This page identifies the published ideas that inform LibTree's design. A cited
algorithm is not a claim that LibTree reproduces the paper's complete method or
its reported speedups. The implementation notes below describe the differences.

## Primary sources

1. Tianqi Chen and Carlos Guestrin. “XGBoost: A Scalable Tree Boosting System.”
   *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
   Discovery and Data Mining*, 2016, pp. 785–794.
   [DOI: 10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785),
   [author preprint](https://arxiv.org/abs/1603.02754).
2. Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma,
   Qiwei Ye, and Tie-Yan Liu. “LightGBM: A Highly Efficient Gradient Boosting
   Decision Tree.” *Advances in Neural Information Processing Systems 30*,
   2017, pp. 3146–3154.
   [NeurIPS paper](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html),
   [author preprint](https://arxiv.org/abs/1711.07230).
3. Google. [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
   This is a style reference, not a performance or algorithmic result.

## What LibTree uses, and what differs

| Published idea | LibTree implementation | Deliberate difference / limit |
| --- | --- | --- |
| Second-order gradient tree scoring, regularized leaf values and missing-value-aware split search [1] | `Stats` stores gradient, Hessian and count; `FindBestSplit` scores prefixes and evaluates both missing directions | Dense `float32` input only; no external-memory blocks, distributed training, GPU, or weighted quantile sketch |
| Cache-conscious data structures and sparsity-aware split processing [1] | Borrowed row-major input, compact `uint16` bins, column-contiguous bins and a high-frequency-bin exception path | The sparse path is an optimization over dense input, not a sparse-matrix API; it does not implement XGBoost's block cache or approximate sketch |
| Histogram construction, histogram subtraction, and feature/data parallel strategies [2] | Build the smaller child's histogram and subtract from the parent for its sibling; reuse bins across boosting rounds; parallelize independent feature/node work when the workload is large enough | One tree's split dependencies remain sequential on the recursive path; histogram subtraction reduces work but does not by itself guarantee parallel speedup |
| Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) [2] | Not implemented | GOSS changes the sampling/weighting approximation; EFB depends on sparse, mutually exclusive features. Both need dedicated quality, memory and runtime ablations before adoption |
| Leaf-wise growth with a leaf limit [2] | Current public learner is depth-limited and uses depth-first/layer-parallel construction | We do not claim LightGBM's leaf-wise model or its quality/runtime tradeoff. A new grow policy needs a separate API, serialization compatibility, and matched-leaf-count study |
| Feature-parallel/data-parallel histogram work [2] | Bounded C++ executor parallelizes feature preparation, eligible histograms, tree levels, derivative updates and prediction batches | Current measurements show LightGBM has better 1→4 fit scaling on the recorded medium/large suite; see [benchmark results](../benchmarks/README.md) |

## Engineering interpretation

The most transferable combination is not “copy both libraries.” It is to keep
the second-order split objective and explicit data locality of XGBoost alongside
LightGBM's compact histogram representation, smaller-child subtraction and
careful parallel work exposure. LibTree applies these with deterministic
accumulation order, RAII-owned scratch, a bounded thread pool, serial fallbacks
for small jobs and a portable CPU-only core.

The measured strengths are narrower and explicit: the historical five-dataset
single-thread snapshot recorded faster LibTree prediction, and the latest
1→4 training scaling exceeds XGBoost's mean on the recorded medium/large suite.
LightGBM still has higher mean training scaling. These observations are not
causal proof for individual techniques, universal rankings, or Kaggle
leaderboard results. Protocols, arithmetic means, variability, versions and
hardware are recorded in [benchmarks/README.md](../benchmarks/README.md) and
[EXPERIMENTS.md](../benchmarks/EXPERIMENTS.md).

## Reproducible research practice

When adopting another idea, first state its mechanism and assumptions, add an
independent behavior test, then measure a one-change ablation on the same
dataset hashes, splits, parameters and CPU budget. Record prediction quality,
fit and inference time, peak memory, source revision and raw repeats. Keep the
change only when its tradeoff is useful; never attribute a combined-bundle win
to one kernel without an isolated ablation.
