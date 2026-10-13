# C5-style classifier: measurements, ablations and validation

The independent C5-style multiclass classifier is implemented. The results below are measured; this is not a compatible C5.0 implementation.

[原始记录](c5-results.json) · [English](C5_REPORT.md) · [中文](C5_REPORT.zh-CN.md) · [方法/引用](../docs/c5.md) · [交互指南](../docs/C5_GUIDE.html)

## 方法 / Method

936 measurements; 6 tasks; seeds 42/2024/7; 3 repetitions per split/configuration; stratified 75/25 train/test; 1 and 4 threads. Arithmetic means are used throughout; raw repeats, population standard deviations and min/max are retained. The three splits overlap and are not nine independent test sets. No test-set tuning.

Fixed budgets: 30 boosting rounds, max depth 4, 64 histogram bins, min leaf rows 5. C5: SAMME rate 1, alpha .001, gain ratio. sklearn tree/SAMME: entropy, ccp_alpha .001; SAMME rate 1. XGBoost/LightGBM: gradient rate .1, L2=1; LightGBM max 16 leaves/tree. Gradient and discrete boosting rates are not directly interchangeable; these are fixed practical settings, not equal optimized accuracy. XGBoost retains its Hessian-based minimum child weight. The sklearn tree has one estimator; multiclass GBDT can build one tree per class per round.

Numeric NaN is preserved for native learners. sklearn inputs use training-fitted median imputation plus missingness indicators; transformation costs are timed. Bank categories are one-hot encoded from training data only, before engine timing. Public bank data excludes call duration (post-outcome information); Pima invalid physiological zeros become NaN. Real-source URLs/SHA-256 are recorded. Bank marketing predicts subscriptions, not credit defaults; it is not a temporal backtest. Synthetic 20k/100k tasks have 20 features and fixed 3% label noise. Wine has three classes.

CPU: AMD EPYC 9V74 80-Core Processor; 5 affinity CPUs; quota 400000 100000. Compiler: g++ (Debian 14.2.0-19) 14.2.0. Python 3.12.14, NumPy 2.5.4, sklearn 1.9.1, XGBoost 3.4.1, LightGBM 4.7.0.

Timing: fresh models in one imported Python process; native engines run in fresh subprocesses. Startup/data-file I/O are outside native timing; Python includes wrapper/validation costs. Engine order is reproducibly shuffled. No concurrent stress/sanitizer job ran during measurements. No dedicated warm-up is discarded; small timings show noticeable noise. Compare engines within the same interface. sklearn SAMME is serial; its four-thread timings are omitted. Native competitor comparisons cover binary tasks; C5 native/Python parity is checked on all tasks, at each run.

## 效果与性能 / Quality and speed

Python, one thread; means across 3 seeds × 3 repeats. Fit/predict are milliseconds. Repeating a seed measures timing, not additional statistical evidence about accuracy. “Rules” counts retained leaves, including gradient trees; this is a structural proxy, not equivalent global explanations across algorithms.

| Task | Engine | Fit ms | Predict ms | Accuracy % | Balanced acc % | AUC | Rules |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| breast_cancer | c5 | 37.759 | 0.191 | 96.27 | 95.87 | 0.9934 | 237.0 |
| breast_cancer | tree | 6.397 | 0.539 | 92.31 | 91.56 | 0.9463 | 8.7 |
| breast_cancer | adaboost | 183.078 | 3.308 | 95.80 | 95.50 | 0.9941 | 325.0 |
| breast_cancer | xgboost | 14.258 | 0.879 | 95.80 | 95.63 | 0.9941 | 272.3 |
| breast_cancer | lightgbm | 9.804 | 0.926 | 96.74 | 96.37 | 0.9921 | 302.0 |
| wine | c5 | 11.505 | 0.297 | 97.04 | 97.53 | 0.9998 | 165.7 |
| wine | tree | 2.220 | 0.384 | 91.11 | 91.60 | 0.9613 | 6.0 |
| wine | adaboost | 52.741 | 3.495 | 97.78 | 98.02 | 0.9990 | 201.7 |
| wine | xgboost | 13.915 | 0.802 | 96.30 | 96.54 | 0.9981 | 459.0 |
| wine | lightgbm | 8.001 | 0.776 | 97.78 | 97.90 | 0.9981 | 439.3 |
| pima | c5 | 7.281 | 0.140 | 75.87 | 72.58 | 0.8353 | 160.3 |
| pima | tree | 3.058 | 0.856 | 72.22 | 66.55 | 0.7938 | 14.0 |
| pima | adaboost | 74.451 | 3.317 | 72.40 | 68.87 | 0.8023 | 317.7 |
| pima | xgboost | 10.142 | 0.824 | 77.60 | 73.57 | 0.8416 | 394.3 |
| pima | lightgbm | 7.648 | 0.887 | 76.22 | 71.81 | 0.8409 | 404.3 |
| bank_marketing | c5 | 831.747 | 2.482 | 89.89 | 60.45 | 0.7960 | 105.7 |
| bank_marketing | tree | 79.425 | 3.786 | 89.96 | 58.51 | 0.7755 | 8.3 |
| bank_marketing | adaboost | 1617.481 | 13.229 | 89.97 | 58.65 | 0.7969 | 93.7 |
| bank_marketing | xgboost | 160.856 | 5.436 | 90.01 | 59.32 | 0.8010 | 469.3 |
| bank_marketing | lightgbm | 137.508 | 10.329 | 90.01 | 59.07 | 0.8012 | 471.3 |
| synthetic_20k | c5 | 141.141 | 1.912 | 89.11 | 89.11 | 0.9538 | 211.0 |
| synthetic_20k | tree | 176.995 | 1.536 | 80.68 | 80.67 | 0.8848 | 16.0 |
| synthetic_20k | adaboost | 5405.945 | 9.677 | 91.07 | 91.07 | 0.9646 | 399.7 |
| synthetic_20k | xgboost | 80.324 | 2.159 | 91.02 | 91.02 | 0.9641 | 478.0 |
| synthetic_20k | lightgbm | 90.867 | 6.974 | 91.07 | 91.06 | 0.9639 | 478.7 |
| synthetic_100k | c5 | 567.306 | 9.113 | 89.96 | 89.96 | 0.9593 | 219.3 |
| synthetic_100k | tree | 1001.986 | 4.327 | 81.97 | 81.98 | 0.8939 | 16.0 |
| synthetic_100k | adaboost | 31429.659 | 35.222 | 91.91 | 91.91 | 0.9673 | 385.7 |
| synthetic_100k | xgboost | 342.922 | 7.489 | 91.20 | 91.20 | 0.9648 | 480.0 |
| synthetic_100k | lightgbm | 426.594 | 26.078 | 91.19 | 91.19 | 0.9648 | 480.0 |

## 线程扩展 / Thread scaling

Speedup = mean time at one thread / mean time at four threads. Small C5 models deliberately use no pool; their fluctuations do not establish parallel gains. Show medium/large tasks only.

| Task | Interface | Engine | Fit speedup | Predict speedup | Fit at 4 threads ms |
| --- | --- | --- | ---: | ---: | ---: |
| bank_marketing | python | c5 | 2.59× | 2.15× | 320.851 |
| bank_marketing | python | xgboost | 0.92× | 1.18× | 174.491 |
| bank_marketing | python | lightgbm | 1.71× | 2.42× | 80.560 |
| bank_marketing | cpp | c5 | 2.44× | 1.60× | 338.964 |
| bank_marketing | cpp | xgboost | 0.95× | 0.95× | 154.450 |
| bank_marketing | cpp | lightgbm | 1.60× | 2.37× | 81.846 |
| synthetic_20k | python | c5 | 1.44× | 2.28× | 98.130 |
| synthetic_20k | python | xgboost | 1.01× | 1.04× | 79.166 |
| synthetic_20k | python | lightgbm | 2.02× | 2.41× | 44.925 |
| synthetic_20k | cpp | c5 | 1.52× | 2.45× | 100.574 |
| synthetic_20k | cpp | xgboost | 0.96× | 0.84× | 76.355 |
| synthetic_20k | cpp | lightgbm | 2.41× | 3.11× | 36.503 |
| synthetic_100k | python | c5 | 1.83× | 2.84× | 310.133 |
| synthetic_100k | python | xgboost | 1.03× | 1.03× | 334.400 |
| synthetic_100k | python | lightgbm | 2.54× | 2.87× | 167.867 |
| synthetic_100k | cpp | c5 | 1.85× | 2.41× | 335.535 |
| synthetic_100k | cpp | xgboost | 1.07× | 1.00× | 324.522 |
| synthetic_100k | cpp | lightgbm | 2.42× | 3.02× | 174.458 |

## 消融 / Ablations

All are one-thread, preselected configurations. `c5_single`: one tree; `c5_unpruned`: alpha=0; `c5_entropy`/`c5_gini`: change only the criterion. All other C5 settings stay fixed. Shorter rules always come from actual pruning/merged bounds, not selective export.

| Task | Variant | Accuracy % | Train accuracy % | AUC | Fit ms | Rules | Conditions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| breast_cancer | c5_single | 94.41 | 97.03 | 0.9362 | 2.089 | 5.7 | 15.3 |
| breast_cancer | c5_unpruned | 96.74 | 100.00 | 0.9916 | 37.492 | 293.7 | 1003.3 |
| breast_cancer | c5 | 96.27 | 100.00 | 0.9934 | 37.759 | 237.0 | 757.3 |
| breast_cancer | c5_entropy | 96.50 | 100.00 | 0.9959 | 35.145 | 267.0 | 873.7 |
| breast_cancer | c5_gini | 96.97 | 100.00 | 0.9936 | 17.659 | 285.7 | 954.7 |
| wine | c5_single | 92.59 | 97.24 | 0.9487 | 0.553 | 5.0 | 12.7 |
| wine | c5_unpruned | 96.30 | 100.00 | 0.9986 | 8.925 | 186.7 | 571.7 |
| wine | c5 | 97.04 | 100.00 | 0.9998 | 11.505 | 165.7 | 433.7 |
| wine | c5_entropy | 98.52 | 100.00 | 0.9998 | 9.544 | 169.0 | 435.0 |
| wine | c5_gini | 98.52 | 100.00 | 0.9995 | 5.093 | 171.3 | 446.3 |
| pima | c5_single | 70.83 | 75.06 | 0.6548 | 1.001 | 7.0 | 17.3 |
| pima | c5_unpruned | 76.56 | 83.45 | 0.8324 | 7.599 | 196.3 | 522.0 |
| pima | c5 | 75.87 | 83.04 | 0.8353 | 7.281 | 160.3 | 393.7 |
| pima | c5_entropy | 74.31 | 96.76 | 0.8153 | 9.114 | 257.3 | 749.3 |
| pima | c5_gini | 71.87 | 99.65 | 0.7975 | 5.820 | 309.7 | 971.0 |
| bank_marketing | c5_single | 89.73 | 89.75 | 0.5866 | 74.574 | 2.0 | 2.0 |
| bank_marketing | c5_unpruned | 89.91 | 90.14 | 0.7966 | 836.581 | 355.3 | 1260.7 |
| bank_marketing | c5 | 89.89 | 89.95 | 0.7960 | 831.747 | 105.7 | 238.3 |
| bank_marketing | c5_entropy | 89.95 | 90.19 | 0.8016 | 841.616 | 175.3 | 495.0 |
| bank_marketing | c5_gini | 89.95 | 90.14 | 0.8023 | 817.795 | 189.0 | 532.7 |
| synthetic_20k | c5_single | 76.23 | 76.50 | 0.7623 | 35.463 | 9.3 | 28.3 |
| synthetic_20k | c5_unpruned | 88.97 | 90.42 | 0.9527 | 145.027 | 407.3 | 1474.3 |
| synthetic_20k | c5 | 89.11 | 90.22 | 0.9538 | 141.141 | 211.0 | 629.3 |
| synthetic_20k | c5_entropy | 91.79 | 93.45 | 0.9685 | 145.417 | 319.0 | 1117.7 |
| synthetic_20k | c5_gini | 91.88 | 93.58 | 0.9665 | 127.850 | 348.3 | 1228.3 |
| synthetic_100k | c5_single | 76.89 | 77.20 | 0.7689 | 183.673 | 10.0 | 31.7 |
| synthetic_100k | c5_unpruned | 89.92 | 90.24 | 0.9584 | 615.926 | 438.7 | 1610.7 |
| synthetic_100k | c5 | 89.96 | 90.27 | 0.9593 | 567.306 | 219.3 | 659.0 |
| synthetic_100k | c5_entropy | 92.04 | 92.42 | 0.9683 | 598.140 | 303.0 | 1041.0 |
| synthetic_100k | c5_gini | 92.35 | 92.77 | 0.9698 | 573.685 | 324.0 | 1132.7 |

## 波动与异常 / Variability and outliers

| Task | Interface | Threads | C5 fit mean ± std ms | Min–max ms | Predict mean ± std ms |
| --- | --- | ---: | ---: | ---: | ---: |
| breast_cancer | python | 1 | 37.759 ± 3.999 | 34.289–45.401 | 0.191 ± 0.046 |
| breast_cancer | python | 4 | 36.312 ± 4.297 | 32.357–47.518 | 0.168 ± 0.009 |
| breast_cancer | cpp | 1 | 36.289 ± 1.594 | 34.185–38.744 | 0.115 ± 0.070 |
| breast_cancer | cpp | 4 | 36.900 ± 3.005 | 32.928–40.906 | 0.092 ± 0.012 |
| wine | python | 1 | 11.505 ± 2.290 | 9.152–16.191 | 0.297 ± 0.441 |
| wine | python | 4 | 10.928 ± 1.018 | 9.590–12.775 | 0.090 ± 0.015 |
| wine | cpp | 1 | 10.722 ± 1.497 | 8.816–12.887 | 0.026 ± 0.002 |
| wine | cpp | 4 | 11.512 ± 1.323 | 9.579–13.876 | 0.035 ± 0.024 |
| pima | python | 1 | 7.281 ± 1.327 | 6.022–10.731 | 0.140 ± 0.043 |
| pima | python | 4 | 7.575 ± 0.899 | 6.093–9.631 | 0.125 ± 0.032 |
| pima | cpp | 1 | 6.923 ± 0.676 | 5.842–8.334 | 0.067 ± 0.004 |
| pima | cpp | 4 | 6.962 ± 0.482 | 6.472–8.039 | 0.086 ± 0.035 |
| bank_marketing | python | 1 | 831.747 ± 63.071 | 758.291–925.976 | 2.482 ± 0.315 |
| bank_marketing | python | 4 | 320.851 ± 14.914 | 298.783–343.256 | 1.152 ± 0.168 |
| bank_marketing | cpp | 1 | 828.677 ± 73.521 | 755.564–999.979 | 2.540 ± 0.325 |
| bank_marketing | cpp | 4 | 338.964 ± 23.953 | 308.356–379.500 | 1.589 ± 0.344 |
| synthetic_20k | python | 1 | 141.141 ± 14.548 | 124.796–178.258 | 1.912 ± 0.229 |
| synthetic_20k | python | 4 | 98.130 ± 8.368 | 85.132–109.095 | 0.837 ± 0.063 |
| synthetic_20k | cpp | 1 | 152.956 ± 13.395 | 133.962–179.986 | 1.801 ± 0.166 |
| synthetic_20k | cpp | 4 | 100.574 ± 10.654 | 86.921–115.183 | 0.734 ± 0.102 |
| synthetic_100k | python | 1 | 567.306 ± 32.552 | 519.445–608.987 | 9.113 ± 0.967 |
| synthetic_100k | python | 4 | 310.133 ± 18.134 | 281.339–346.667 | 3.203 ± 0.237 |
| synthetic_100k | cpp | 1 | 619.623 ± 43.141 | 563.103–699.668 | 9.130 ± 0.688 |
| synthetic_100k | cpp | 4 | 335.535 ± 15.655 | 311.117–362.728 | 3.783 ± 1.058 |

## 结论与限制 / Conclusions and limits

Default gain ratio favors compact rules but does not lead every task. No universal scaling/top1 claim is supported.

Bank pruning: rules 355.3 → 105.7; accuracy 89.91% → 89.89%.

Python bank_marketing scaling: c5: 2.59× fit, 2.15× predict; xgboost: 0.92× fit, 1.18× predict; lightgbm: 1.71× fit, 2.42× predict.

Python synthetic_100k scaling: c5: 1.83× fit, 2.84× predict; xgboost: 1.03× fit, 1.03× predict; lightgbm: 2.54× fit, 2.87× predict.

synthetic_20k accuracy: c5_gini: 91.88%; c5: 89.11%; xgboost: 91.02%; lightgbm: 91.07%.

synthetic_100k accuracy: c5_gini: 92.35%; c5: 89.96%; xgboost: 91.20%; lightgbm: 91.19%.

pima accuracy: c5_gini: 71.87%; c5: 75.87%; xgboost: 77.60%; lightgbm: 76.22%.

Gini improves the synthetic tasks, with more rules, but worsens Pima. Switching defaults based on these tests needs fresh validation. Small-data training accuracy near 100% warrants cross-validation and stronger simplification. Bank imbalance makes balanced accuracy/AUC essential.

C5 probabilities use a conservative, uncalibrated normalized-vote softmax, unlike competitor probability transforms. Log loss is retained in raw summaries; calibration is not an advantage claim. Rule counts/merged condition counts are structural proxies; no human comprehension study, official C5.0 comparison, Kaggle leaderboard submission or temporal financial backtest was performed.

## TDD 与验证 / TDD and validation

Tests first failed because C5Classifier did not exist; implementation then made them pass. Independent rule evaluators reproduce predictions/votes; tests cover multiclass, NaN routing, sample weights, pruning, early stop, ties, refit/ownership, errors, thread parity and concurrent readers. Allocation fault injection checks failed refits preserve models. No new coverage exclusion was added.

Release CTest 7/7; Python estimator/inspection/C5 68/68; CLI 16/16; performance/RSS 3/3; benchmark methodology 12/12. ASan/UBSan/LSan: CTest 7/7 and CLI 16/16; TSan CTest 7/7. Native instrumented source lines 2434/2434 (100%); Python package statements 359/359 (100%). CI enforces both 100% gates. Coverage scope: src/*.cc and internal headers, plus python/libtree; benchmark harnesses and interactive guide JavaScript are outside these gates. Existing documented platform/exhaustion exclusions remain. Coverage is not proof of absence of bugs.

## 复现 / Reproduce

```sh
python -m pip install '.[dev,benchmark]'
make test
make sanitize
make thread-sanitize
make coverage
PYTHONPATH=python:. python benchmarks/c5.py --seeds 42 2024 7 --threads 1 4 \
  --repeats 3 --rounds 30 --depth 4 --output benchmarks/c5-results.json
PYTHONPATH=python:. python benchmarks/c5_report.py
```


Measured implementation/source SHA-256: `873243a5a2e8299e097e0656409428002b148210fc27e5685ea0a175f96ad146`. Downloaded data/builds/temporary predictions are ignored by git; raw results, report scripts and these reports are tracked. Complete per-group arithmetic mean/std/min/max values and per-run metrics remain in the JSON.
