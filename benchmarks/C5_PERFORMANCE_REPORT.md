# C5: faster training, direct class prediction and tree inspection

[English](C5_PERFORMANCE_REPORT.md) · [中文](C5_PERFORMANCE_REPORT.zh-CN.md) · [raw measurements](c5-performance-results.json) · [guide and references](../docs/c5.md)

## Scope and method

972 interface measurements on 7 tasks; seeds [42, 2024, 7]; 3 timed fresh models per split/configuration; threads [1, 4]. Arithmetic means, population standard deviations and extrema are retained. Repeated seeds add timing evidence, not independent test sets; the three holdouts overlap.

Baseline revision: `60f003dfa9ce316c39142c881ad372121e4ff3b8`. C5 defaults, split criteria, round budget and pruning are unchanged. Both static libraries use the same Release benchmark-driver source. Each Python worker imports its own checkout wrapper/library; engines run in reproducibly shuffled blocks, never concurrently. Fit, probability prediction, then class prediction are timed separately. Startup, imports and input I/O are excluded; wrapper validation is included. Class timing follows probability timing and benefits from warm caches; no dedicated warmup is discarded.

The first Python fit in each worker also includes lazy native-library loading; subsequent repetitions reuse that loaded library. This matters most for short tasks.

C5: up to 30 SAMME rounds, depth 4, 64 bins, minimum leaf rows 5, rate 1, alpha=.001, gain ratio. Competitors use the same round/depth/bin budgets, gradient rate .1 and L2=1; LightGBM has at most 16 leaves at depth 4. Algorithms and pruning units differ; this is fixed-budget comparison, not equal tuning. Native competitor comparisons cover binary tasks; Wine is multiclass, and gradient boosting may build a tree per class per round. Native competitor label timings are unavailable. Compare engines within an interface; native XGBoost probability timing includes test DMatrix construction.

Round budgets are not equal tree counts in multiclass tasks; Python timings include label encoding and wrapper checks during fitting, whereas native inputs are already prepared. C5 can stop on perfect or weak learners; accepted estimator counts are retained per run.

Stratified 75/25 holdouts; categorical encoders fit training rows only. Source URLs/checksums and split hashes are in JSON. German credit and credit status use bad-credit labels; Taiwan uses September late payment with April features, not the original full default target. Random splits are not financial time backtests. Synthetic tasks have 3% label noise; missing_20k adds a fixed 10% NaN mask. No test-set tuning was used.

CPU AMD EPYC 9V74 80-Core Processor; 5 affinity CPUs; quota 400000 100000; c++ (Debian 14.2.0-19) 14.2.0; Python 3.12.14; NumPy 2.5.4; sklearn 1.9.1; XGBoost 3.4.1; LightGBM 4.7.0.

| Task | Train rows | Test rows | Features in first split |
| --- | ---: | ---: | ---: |
| wine | 133 | 45 | 13 |
| german_credit | 750 | 250 | 61 |
| credit_status | 3340 | 1114 | 29 |
| taiwan_late_payment | 22500 | 7500 | 3 |
| synthetic_20k | 15000 | 5000 | 20 |
| synthetic_100k | 75000 | 25000 | 20 |
| missing_20k | 15000 | 5000 | 20 |

## Implemented changes

Feature-local histogram workspaces persist across nodes/rounds; parent impurity is computed once per node. Without missing rows, duplicate missing-direction candidates are skipped, preserving left-first ties. Retained leaf labels are reused for SAMME weighted errors. Prediction shares validation/traversal, avoids duplicate native validation and exp(0), and Python class prediction calls native class voting directly. Scratch remains call-local. One padded allocation replaces per-worker vote-buffer allocations; 64-byte gaps avoid sharing 64-byte cache lines, and static row ranges balance small/deep ensembles. Histogram workspace adds temporary O(features × bins × classes) storage during fitting. See the cited data-locality motivation in the guide; C5 does not implement histogram subtraction.

## Before/after training

Speedup = before mean / after mean; below 1 means slower. These are bundle-level A/B results, not isolated causal attribution to each optimization.

| Task | Interface | Threads | Before ms | After mean ± std ms | After min–max ms | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| wine | python | 1 | 9.944 | 6.201 ± 0.562 | 5.550–7.653 | 1.60× |
| wine | python | 4 | 9.974 | 6.321 ± 1.250 | 5.287–9.714 | 1.58× |
| wine | cpp | 1 | 9.538 | 5.879 ± 0.710 | 5.055–7.533 | 1.62× |
| wine | cpp | 4 | 9.646 | 5.936 ± 0.463 | 5.471–7.023 | 1.62× |
| german_credit | python | 1 | 17.491 | 15.680 ± 0.446 | 15.238–16.569 | 1.12× |
| german_credit | python | 4 | 17.129 | 15.595 ± 0.517 | 14.877–16.375 | 1.10× |
| german_credit | cpp | 1 | 16.944 | 15.389 ± 0.501 | 14.733–16.488 | 1.10× |
| german_credit | cpp | 4 | 17.492 | 15.367 ± 0.382 | 14.920–16.189 | 1.14× |
| credit_status | python | 1 | 37.951 | 35.896 ± 1.201 | 34.390–38.513 | 1.06× |
| credit_status | python | 4 | 28.815 | 24.832 ± 2.163 | 22.105–28.343 | 1.16× |
| credit_status | cpp | 1 | 39.272 | 36.965 ± 2.072 | 34.873–42.340 | 1.06× |
| credit_status | cpp | 4 | 30.761 | 25.184 ± 1.991 | 23.079–28.471 | 1.22× |
| taiwan_late_payment | python | 1 | 45.202 | 41.587 ± 4.778 | 33.612–46.643 | 1.09× |
| taiwan_late_payment | python | 4 | 46.996 | 42.282 ± 7.102 | 31.608–52.476 | 1.11× |
| taiwan_late_payment | cpp | 1 | 49.323 | 44.480 ± 6.488 | 32.235–52.654 | 1.11× |
| taiwan_late_payment | cpp | 4 | 47.986 | 49.387 ± 14.535 | 31.298–69.570 | 0.97× |
| synthetic_20k | python | 1 | 128.390 | 118.707 ± 4.185 | 112.210–126.104 | 1.08× |
| synthetic_20k | python | 4 | 88.630 | 70.869 ± 4.638 | 65.639–80.562 | 1.25× |
| synthetic_20k | cpp | 1 | 132.222 | 127.810 ± 19.713 | 113.693–179.862 | 1.03× |
| synthetic_20k | cpp | 4 | 89.982 | 75.436 ± 3.609 | 69.506–80.583 | 1.19× |
| synthetic_100k | python | 1 | 549.783 | 545.042 ± 30.320 | 499.864–596.030 | 1.01× |
| synthetic_100k | python | 4 | 298.921 | 276.674 ± 20.443 | 241.127–303.298 | 1.08× |
| synthetic_100k | cpp | 1 | 563.153 | 564.016 ± 27.720 | 529.168–603.223 | 1.00× |
| synthetic_100k | cpp | 4 | 310.471 | 302.020 ± 26.652 | 257.470–333.756 | 1.03× |
| missing_20k | python | 1 | 136.033 | 132.557 ± 9.569 | 126.685–157.038 | 1.03× |
| missing_20k | python | 4 | 92.073 | 87.969 ± 4.749 | 82.461–99.229 | 1.05× |
| missing_20k | cpp | 1 | 137.478 | 138.266 ± 4.671 | 131.718–144.298 | 0.99× |
| missing_20k | cpp | 4 | 92.006 | 89.673 ± 4.692 | 81.822–96.152 | 1.03× |

## Before/after inference

| Task | Interface | Threads | Before probability ms | After probability ms | Speedup | Before class ms | After class ms | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wine | python | 1 | 0.058 | 0.059 | 0.99× | 0.032 | 0.035 | 0.92× |
| wine | python | 4 | 0.061 | 0.057 | 1.06× | 0.032 | 0.041 | 0.77× |
| wine | cpp | 1 | 0.025 | 0.027 | 0.93× | 0.012 | 0.012 | 0.97× |
| wine | cpp | 4 | 0.025 | 0.025 | 0.98× | 0.012 | 0.014 | 0.89× |
| german_credit | python | 1 | 0.144 | 0.153 | 0.94× | 0.090 | 0.101 | 0.89× |
| german_credit | python | 4 | 0.146 | 0.141 | 1.03× | 0.096 | 0.092 | 1.04× |
| german_credit | cpp | 1 | 0.111 | 0.096 | 1.15× | 0.068 | 0.070 | 0.97× |
| german_credit | cpp | 4 | 0.105 | 0.102 | 1.03× | 0.066 | 0.071 | 0.93× |
| credit_status | python | 1 | 0.402 | 0.437 | 0.92× | 0.333 | 0.336 | 0.99× |
| credit_status | python | 4 | 0.440 | 0.423 | 1.04× | 0.385 | 0.362 | 1.07× |
| credit_status | cpp | 1 | 0.370 | 0.396 | 0.93× | 0.301 | 0.299 | 1.01× |
| credit_status | cpp | 4 | 0.375 | 0.347 | 1.08× | 0.294 | 0.280 | 1.05× |
| taiwan_late_payment | python | 1 | 1.280 | 1.274 | 1.00× | 1.257 | 1.134 | 1.11× |
| taiwan_late_payment | python | 4 | 0.577 | 0.651 | 0.89× | 0.585 | 0.795 | 0.74× |
| taiwan_late_payment | cpp | 1 | 1.148 | 1.183 | 0.97× | 1.048 | 1.034 | 1.01× |
| taiwan_late_payment | cpp | 4 | 0.711 | 0.676 | 1.05× | 0.450 | 0.408 | 1.10× |
| synthetic_20k | python | 1 | 1.946 | 1.847 | 1.05× | 1.864 | 1.564 | 1.19× |
| synthetic_20k | python | 4 | 0.734 | 0.731 | 1.00× | 0.680 | 0.627 | 1.08× |
| synthetic_20k | cpp | 1 | 1.707 | 1.637 | 1.04× | 1.439 | 1.508 | 0.95× |
| synthetic_20k | cpp | 4 | 0.671 | 0.662 | 1.01× | 0.566 | 0.612 | 0.92× |
| synthetic_100k | python | 1 | 8.692 | 8.566 | 1.01× | 8.665 | 7.930 | 1.09× |
| synthetic_100k | python | 4 | 3.542 | 2.958 | 1.20× | 3.395 | 2.641 | 1.29× |
| synthetic_100k | cpp | 1 | 9.117 | 8.550 | 1.07× | 7.984 | 8.147 | 0.98× |
| synthetic_100k | cpp | 4 | 3.726 | 3.371 | 1.11× | 2.715 | 2.872 | 0.95× |
| missing_20k | python | 1 | 2.797 | 2.870 | 0.97× | 2.597 | 2.374 | 1.09× |
| missing_20k | python | 4 | 1.006 | 1.019 | 0.99× | 0.837 | 0.935 | 0.90× |
| missing_20k | cpp | 1 | 2.551 | 2.577 | 0.99× | 2.456 | 2.271 | 1.08× |
| missing_20k | cpp | 4 | 0.924 | 0.892 | 1.04× | 0.776 | 0.761 | 1.02× |

## Inference variability

| Task | Interface | Threads | After probability mean ± std ms | Probability min–max ms | After class mean ± std ms | Class min–max ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| wine | python | 1 | 0.059 ± 0.012 | 0.049–0.090 | 0.035 ± 0.006 | 0.028–0.048 |
| wine | python | 4 | 0.057 ± 0.022 | 0.042–0.119 | 0.041 ± 0.028 | 0.025–0.121 |
| wine | cpp | 1 | 0.027 ± 0.005 | 0.023–0.039 | 0.012 ± 0.001 | 0.011–0.015 |
| wine | cpp | 4 | 0.025 ± 0.001 | 0.023–0.028 | 0.014 ± 0.004 | 0.011–0.025 |
| german_credit | python | 1 | 0.153 ± 0.014 | 0.136–0.174 | 0.101 ± 0.015 | 0.090–0.132 |
| german_credit | python | 4 | 0.141 ± 0.018 | 0.129–0.192 | 0.092 ± 0.004 | 0.086–0.099 |
| german_credit | cpp | 1 | 0.096 ± 0.004 | 0.092–0.103 | 0.070 ± 0.007 | 0.065–0.085 |
| german_credit | cpp | 4 | 0.102 ± 0.011 | 0.092–0.124 | 0.071 ± 0.008 | 0.065–0.090 |
| credit_status | python | 1 | 0.437 ± 0.033 | 0.403–0.510 | 0.336 ± 0.030 | 0.312–0.391 |
| credit_status | python | 4 | 0.423 ± 0.039 | 0.384–0.502 | 0.362 ± 0.086 | 0.299–0.593 |
| credit_status | cpp | 1 | 0.396 ± 0.127 | 0.319–0.745 | 0.299 ± 0.034 | 0.261–0.370 |
| credit_status | cpp | 4 | 0.347 ± 0.023 | 0.317–0.387 | 0.280 ± 0.017 | 0.260–0.318 |
| taiwan_late_payment | python | 1 | 1.274 ± 0.096 | 1.156–1.401 | 1.134 ± 0.177 | 0.948–1.518 |
| taiwan_late_payment | python | 4 | 0.651 ± 0.186 | 0.420–1.007 | 0.795 ± 0.764 | 0.413–2.944 |
| taiwan_late_payment | cpp | 1 | 1.183 ± 0.166 | 0.977–1.453 | 1.034 ± 0.145 | 0.846–1.294 |
| taiwan_late_payment | cpp | 4 | 0.676 ± 0.595 | 0.354–2.342 | 0.408 ± 0.074 | 0.305–0.569 |
| synthetic_20k | python | 1 | 1.847 ± 0.198 | 1.557–2.156 | 1.564 ± 0.159 | 1.366–1.808 |
| synthetic_20k | python | 4 | 0.731 ± 0.086 | 0.594–0.887 | 0.627 ± 0.042 | 0.534–0.681 |
| synthetic_20k | cpp | 1 | 1.637 ± 0.083 | 1.503–1.782 | 1.508 ± 0.116 | 1.321–1.753 |
| synthetic_20k | cpp | 4 | 0.662 ± 0.026 | 0.614–0.702 | 0.612 ± 0.065 | 0.547–0.764 |
| synthetic_100k | python | 1 | 8.566 ± 0.361 | 8.132–9.357 | 7.930 ± 0.321 | 7.373–8.615 |
| synthetic_100k | python | 4 | 2.958 ± 0.331 | 2.453–3.480 | 2.641 ± 0.319 | 2.197–3.065 |
| synthetic_100k | cpp | 1 | 8.550 ± 0.468 | 7.946–9.353 | 8.147 ± 0.631 | 7.226–9.490 |
| synthetic_100k | cpp | 4 | 3.371 ± 0.820 | 2.584–5.520 | 2.872 ± 0.252 | 2.348–3.269 |
| missing_20k | python | 1 | 2.870 ± 0.636 | 2.457–4.613 | 2.374 ± 0.167 | 2.180–2.614 |
| missing_20k | python | 4 | 1.019 ± 0.084 | 0.929–1.188 | 0.935 ± 0.145 | 0.808–1.308 |
| missing_20k | cpp | 1 | 2.577 ± 0.236 | 2.272–3.038 | 2.271 ± 0.174 | 2.039–2.542 |
| missing_20k | cpp | 4 | 0.892 ± 0.090 | 0.772–1.051 | 0.761 ± 0.092 | 0.612–0.935 |

## Held-out quality and competitor speed

Python, one thread, arithmetic means. Before/after C5 rule JSON, training probabilities and test probabilities are byte-identical within each interface, seed and split, including across thread counts/repeats. Thus measured C5 accuracy/AUC/KS are unchanged. KS is absolute bad/good score-distribution distance; directional KS and AP are also retained. Wine has no binary KS/AP. Scores are uncalibrated; these are not production risk probabilities.

| Task | Model | Fit ms | Probability ms | Accuracy % | Balanced acc % | AUC | KS | AP | Rules |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wine | after | 6.201 | 0.059 | 97.04 | 97.53 | 0.9998 | — | — | 165.7 |
| wine | xgboost | 11.346 | 0.591 | 96.30 | 96.54 | 0.9981 | — | — | 459.0 |
| wine | lightgbm | 7.516 | 0.516 | 97.78 | 97.90 | 0.9981 | — | — | 439.3 |
| german_credit | after | 15.680 | 0.153 | 75.47 | 69.02 | 0.7750 | 0.4800 | 0.5781 | 217.3 |
| german_credit | xgboost | 9.407 | 0.616 | 74.27 | 64.60 | 0.7925 | 0.4781 | 0.6170 | 427.0 |
| german_credit | lightgbm | 10.130 | 0.864 | 74.13 | 64.63 | 0.7857 | 0.4927 | 0.5994 | 432.3 |
| credit_status | after | 35.896 | 0.437 | 79.80 | 72.75 | 0.8267 | 0.5392 | 0.6662 | 202.7 |
| credit_status | xgboost | 19.424 | 0.988 | 78.99 | 68.99 | 0.8377 | 0.5380 | 0.6711 | 468.3 |
| credit_status | lightgbm | 18.032 | 1.691 | 79.08 | 68.96 | 0.8389 | 0.5504 | 0.6735 | 470.0 |
| taiwan_late_payment | after | 41.587 | 1.274 | 79.86 | 61.42 | 0.6836 | 0.2925 | 0.4323 | 87.3 |
| taiwan_late_payment | xgboost | 30.282 | 2.374 | 79.82 | 60.69 | 0.6822 | 0.2918 | 0.4340 | 466.7 |
| taiwan_late_payment | lightgbm | 33.698 | 4.500 | 79.73 | 60.44 | 0.6823 | 0.2911 | 0.4320 | 465.0 |
| synthetic_20k | after | 118.707 | 1.847 | 89.11 | 89.11 | 0.9538 | 0.7853 | 0.9505 | 211.0 |
| synthetic_20k | xgboost | 74.821 | 2.137 | 91.02 | 91.02 | 0.9641 | 0.8235 | 0.9598 | 478.0 |
| synthetic_20k | lightgbm | 85.272 | 6.026 | 91.07 | 91.06 | 0.9639 | 0.8223 | 0.9590 | 478.7 |
| synthetic_100k | after | 545.042 | 8.566 | 89.96 | 89.96 | 0.9593 | 0.8021 | 0.9548 | 219.3 |
| synthetic_100k | xgboost | 326.708 | 7.048 | 91.20 | 91.20 | 0.9648 | 0.8253 | 0.9612 | 480.0 |
| synthetic_100k | lightgbm | 396.623 | 23.237 | 91.19 | 91.19 | 0.9648 | 0.8257 | 0.9608 | 480.0 |
| missing_20k | after | 132.557 | 2.870 | 87.67 | 87.66 | 0.9436 | 0.7561 | 0.9410 | 229.0 |
| missing_20k | xgboost | 91.153 | 3.221 | 88.47 | 88.47 | 0.9492 | 0.7724 | 0.9460 | 479.3 |
| missing_20k | lightgbm | 92.148 | 7.892 | 88.47 | 88.46 | 0.9504 | 0.7767 | 0.9476 | 480.0 |

## Thread scaling

Each engine uses its own one-thread mean as baseline. Small C5 fits may create no workers; fluctuations there are not parallel gains. All ratios, including regressions, are retained.

| Task | Interface | Model | Fit 1→4 speedup | Probability 1→4 speedup | Class 1→4 speedup |
| --- | --- | --- | ---: | ---: | ---: |
| wine | python | before | 1.00× | 0.96× | 1.01× |
| wine | python | after | 0.98× | 1.03× | 0.84× |
| wine | python | xgboost | 1.05× | 0.96× | 0.99× |
| wine | python | lightgbm | 0.85× | 1.02× | 0.88× |
| wine | cpp | before | 0.99× | 1.01× | 1.00× |
| wine | cpp | after | 0.99× | 1.07× | 0.92× |
| german_credit | python | before | 1.02× | 0.98× | 0.94× |
| german_credit | python | after | 1.01× | 1.09× | 1.10× |
| german_credit | python | xgboost | 1.01× | 1.02× | 1.06× |
| german_credit | python | lightgbm | 0.85× | 1.12× | 1.27× |
| german_credit | cpp | before | 0.97× | 1.05× | 1.03× |
| german_credit | cpp | after | 1.00× | 0.94× | 0.99× |
| german_credit | cpp | xgboost | 0.93× | 0.94× | — |
| german_credit | cpp | lightgbm | 0.66× | 1.89× | — |
| credit_status | python | before | 1.32× | 0.91× | 0.87× |
| credit_status | python | after | 1.45× | 1.03× | 0.93× |
| credit_status | python | xgboost | 0.98× | 0.99× | 0.89× |
| credit_status | python | lightgbm | 1.20× | 1.76× | 1.73× |
| credit_status | cpp | before | 1.28× | 0.99× | 1.02× |
| credit_status | cpp | after | 1.47× | 1.14× | 1.07× |
| credit_status | cpp | xgboost | 1.06× | 1.09× | — |
| credit_status | cpp | lightgbm | 1.31× | 2.71× | — |
| taiwan_late_payment | python | before | 0.96× | 2.22× | 2.15× |
| taiwan_late_payment | python | after | 0.98× | 1.96× | 1.43× |
| taiwan_late_payment | python | xgboost | 1.04× | 0.97× | 1.05× |
| taiwan_late_payment | python | lightgbm | 1.39× | 1.79× | 1.73× |
| taiwan_late_payment | cpp | before | 1.03× | 1.61× | 2.33× |
| taiwan_late_payment | cpp | after | 0.90× | 1.75× | 2.54× |
| taiwan_late_payment | cpp | xgboost | 1.02× | 1.05× | — |
| taiwan_late_payment | cpp | lightgbm | 1.49× | 2.24× | — |
| synthetic_20k | python | before | 1.45× | 2.65× | 2.74× |
| synthetic_20k | python | after | 1.68× | 2.53× | 2.49× |
| synthetic_20k | python | xgboost | 0.99× | 1.04× | 1.06× |
| synthetic_20k | python | lightgbm | 2.01× | 2.34× | 2.32× |
| synthetic_20k | cpp | before | 1.47× | 2.54× | 2.54× |
| synthetic_20k | cpp | after | 1.69× | 2.47× | 2.46× |
| synthetic_20k | cpp | xgboost | 1.02× | 0.89× | — |
| synthetic_20k | cpp | lightgbm | 2.10× | 2.81× | — |
| synthetic_100k | python | before | 1.84× | 2.45× | 2.55× |
| synthetic_100k | python | after | 1.97× | 2.90× | 3.00× |
| synthetic_100k | python | xgboost | 0.99× | 0.99× | 1.00× |
| synthetic_100k | python | lightgbm | 2.22× | 2.74× | 2.67× |
| synthetic_100k | cpp | before | 1.81× | 2.45× | 2.94× |
| synthetic_100k | cpp | after | 1.87× | 2.54× | 2.84× |
| synthetic_100k | cpp | xgboost | 1.02× | 1.05× | — |
| synthetic_100k | cpp | lightgbm | 2.42× | 2.97× | — |
| missing_20k | python | before | 1.48× | 2.78× | 3.10× |
| missing_20k | python | after | 1.51× | 2.82× | 2.54× |
| missing_20k | python | xgboost | 0.97× | 0.97× | 1.05× |
| missing_20k | python | lightgbm | 1.98× | 2.64× | 2.47× |
| missing_20k | cpp | before | 1.49× | 2.76× | 3.16× |
| missing_20k | cpp | after | 1.54× | 2.89× | 2.98× |
| missing_20k | cpp | xgboost | 1.08× | 1.12× | — |
| missing_20k | cpp | lightgbm | 2.21× | 3.15× | — |

## Tree inspection cost

On-demand Python one-thread means, outside fit/predict timing. JSON export includes all retained trees; one-sample explanation includes actual paths and votes. Both include C ABI size queries, UTF-8 handling and Python JSON work. They are inspection formats, not model persistence.

| Task | Full JSON export ms | One-sample explanation ms | JSON bytes |
| --- | ---: | ---: | ---: |
| wine | 1.437 | 1.043 | 54723 |
| german_credit | 1.735 | 1.148 | 72143 |
| credit_status | 1.629 | 1.179 | 67190 |
| taiwan_late_payment | 0.727 | 0.716 | 27990 |
| synthetic_20k | 1.790 | 1.438 | 73752 |
| synthetic_100k | 1.933 | 1.480 | 77322 |
| missing_20k | 1.895 | 1.405 | 80452 |

## Findings and limitations

Training improves in 25/28 before/after groups; observed training speedups range from 0.97× to 1.62×. Inspect each inference row separately: reducing training work does not guarantee faster inference or better thread scaling on every task. Quality did not improve because model construction semantics and parameters were preserved.

Shared cloud CPU scheduling, short-task overhead and overlapping splits limit generalization. No statistical-significance, universal winner or human-comprehension claim follows from these timings. Rule counts are leaf-count proxies across different algorithms. C5 probabilities remain uncalibrated, and competitor learning-rate/regularization semantics differ. Raw means/std/min/max and negative outcomes remain available.

The [first candidate raw runs](c5-performance-first-pass.json) are retained, including a 4.03 ms Python four-thread class-prediction spike on missing_20k. That candidate used separate worker buffers and 64-row tiles. The final candidate uses padded shared storage and static row ranges. [This patch](c5-first-pass.patch), applied to this measured source snapshot in a separate checkout, reconstructs the first candidate. These are separate measurement phases, not a randomized isolated ablation; their difference cannot be attributed solely to a kernel. The final allocation-budget test intentionally rejects the first candidate.

## Validation

TDD: tree/path export and direct-class tests failed before implementation; the CLI initially rejected trees, and the one-allocation contract failed with std::bad_alloc. Independent JSON-tree traversal now reproduces actual paths, rules and votes for all four split criteria, including multiclass/NaN. Tests cover constant models, 256 classes, failed refits, output buffers, concurrent inspection/readers and thread-count parity.

Native CTest 7/7; Python estimator/inspection tests 77/77; CLI 16/16; performance/RSS 3/3; methodology 20/20. ASan/UBSan/LSan and TSan passed. Native line coverage 2614/2614; Python statement coverage 393/393, both 100%. No new coverage exclusions. The benchmark harness and report generator are outside production coverage gates.

## Reproduce

```sh
python -m pip install '.[dev,benchmark]'
make all
git worktree add --detach /tmp/libtree-c5-before 60f003d
cmake -S /tmp/libtree-c5-before -B /tmp/libtree-c5-before/build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/libtree-c5-before/build -j4
PYTHONPATH=python:. python benchmarks/c5_performance.py --baseline /tmp/libtree-c5-before \
  --seeds 42 2024 7 --threads 1 4 --repeats 3 --rounds 30 \
  --output benchmarks/c5-performance-results.json
PYTHONPATH=python:. python benchmarks/c5_performance_report.py
make test
make coverage
make sanitize
make thread-sanitize
```

Measured implementation/harness SHA-256: `75cd6ec447668803cc79e4992918dd790924cef15a066786719d3b6943f354ce`. The JSON retains baseline revision, driver binary hashes, source URLs/checksums, split hashes and raw per-run fingerprints. Downloaded/prepared data, predictions and local drivers remain ignored by git.
