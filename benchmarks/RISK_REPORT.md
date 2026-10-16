# C5 risk study: boosting, Hellinger and KS

[English](RISK_REPORT.md) · [中文](RISK_REPORT.zh-CN.md) · [raw runs and CV scores](risk-results.json) · [methods/citations](../docs/c5-risk.md)

## Method

1260 timed interface measurements, 360 inner-CV fits; seeds [42, 2024, 7], 3 repeats per configuration/split, threads [1, 4]. Arithmetic means, population standard deviations and min/max; repeated timings do not add independent test sets. Raw individual runs are retained.

Outer stratified 75/25 holdouts; train-only stratified CV, with encoders refit per inner fold. Select by mean directional KS, then AUC, then fewer rules. Require .01 KS gain over samme_30 or keep baseline. Refit on complete outer training data; evaluate outer test only afterward. No test-set choice of configurations/thresholds. The ten candidates are declared before the run.

Depth 4, bins 64, minimum leaf rows 5. samme_30 uses 30 rounds/rate 1; long variants use 100 rounds. Shrinkage uses rate .3; balancing uses inverse-frequency weights; default alpha=.001, less_pruning=.0002. Competitors use the same maximum round budget, rate 1 for sklearn SAMME and .1 for gradient boosters; L2=1 for gradient boosters. Learning rates/pruning units differ by algorithm. The selected C5 has an additional search budget; its search cost is shown separately.

Python 3.12.14; sklearn 1.9.1; XGBoost 3.4.1; LightGBM 4.7.0; CPU quota 400000 100000. Python times include wrappers/validation. Native timings exclude process startup/file I/O. Fresh models; reproducibly shuffled order; no simultaneous stress/coverage build during measurements. Preprocessing for outer comparisons is common and outside fit timing, except sklearn imputation, which is included. Native/Python probability parity is checked for every non-sklearn measured model. sklearn SAMME is serial and has no standalone C++ comparison.

KS=max absolute empirical bad/good score-distribution difference. Directional KS=max(TPR-FPR); tied scores are grouped. An inverted model can have high absolute KS, so AUC and directional KS must accompany it. Label 1 means bad. These are descriptive held-out statistics; no deployed threshold is picked on the test data. Probabilities are not calibrated; bad recall uses the model argmax/ordinary class decision. Single C5 trees use hard leaf votes, not leaf class-frequency probabilities; their ranking can be constant even with multiple leaves.

## Data

| Dataset | Rows | Bad fraction | Target/source |
| --- | ---: | ---: | --- |
| german_credit | 1000 | 30.00% | bad credit risk; source credit_risk=0 (300/1000) |
| credit_status | 4454 | 28.15% | Status=bad; missing predictors retained; identifier removed |
| taiwan_late_payment | 30000 | 22.73% | late September 2005 payment; predictors April late flag/bill and age; not the original default target |
| rare_risk_20k | 20000 | 3.50% | explicitly synthetic rare-risk classification |

Taiwan is a real public subset with April features and September late-payment labels; it is not the original full default target. IDs and targets are excluded. Sources and SHA-256 are pinned. Random splits are not temporal backtests or external-lender validation.

## Quality and performance

One-thread Python, arithmetic means; fit/predict in milliseconds. selected uses only training CV. The fixed Hellinger row is an ablation, not a configuration selected using these test scores.

| Task | Model | KS | AUC | AP | Bad recall | Balanced accuracy | Fit ms | Predict ms | Rules |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| german_credit | single | 0.0305 | 0.5152 | 0.3103 | 0.0667 | 0.5152 | 1.729 | 0.043 | 6.7 |
| german_credit | samme_30 | 0.4800 | 0.7750 | 0.5781 | 0.5289 | 0.6902 | 17.825 | 0.176 | 217.3 |
| german_credit | hellinger | 0.4971 | 0.8009 | 0.6206 | 0.5467 | 0.7029 | 58.153 | 0.535 | 818.7 |
| german_credit | selected | 0.4629 | 0.7942 | 0.6239 | 0.6267 | 0.7057 | 56.676 | 0.540 | 863.0 |
| german_credit | adaboost | 0.4552 | 0.7757 | 0.6131 | 0.5644 | 0.7041 | 283.291 | 8.649 | 1198.7 |
| german_credit | xgboost | 0.4686 | 0.7928 | 0.6240 | 0.4978 | 0.6803 | 20.522 | 0.962 | 1244.0 |
| german_credit | lightgbm | 0.4616 | 0.7910 | 0.6191 | 0.5111 | 0.6879 | 20.121 | 1.691 | 1289.0 |
| credit_status | single | 0.3868 | 0.6934 | 0.4399 | 0.5510 | 0.6934 | 4.379 | 0.111 | 7.3 |
| credit_status | samme_30 | 0.5392 | 0.8267 | 0.6662 | 0.5658 | 0.7275 | 40.941 | 0.463 | 202.7 |
| credit_status | hellinger | 0.5595 | 0.8436 | 0.6911 | 0.5541 | 0.7275 | 122.108 | 1.800 | 743.3 |
| credit_status | selected | 0.5541 | 0.8419 | 0.6906 | 0.5913 | 0.7309 | 129.510 | 1.675 | 685.0 |
| credit_status | adaboost | 0.5258 | 0.8208 | 0.6439 | 0.5446 | 0.7127 | 650.814 | 8.932 | 706.7 |
| credit_status | xgboost | 0.5508 | 0.8396 | 0.6790 | 0.5435 | 0.7180 | 52.510 | 1.987 | 1439.7 |
| credit_status | lightgbm | 0.5456 | 0.8401 | 0.6824 | 0.5340 | 0.7166 | 39.866 | 4.704 | 1461.0 |
| taiwan_late_payment | single | 0.2301 | 0.6151 | 0.3389 | 0.2787 | 0.6151 | 7.447 | 0.373 | 2.0 |
| taiwan_late_payment | samme_30 | 0.2925 | 0.6836 | 0.4323 | 0.2762 | 0.6142 | 45.354 | 1.375 | 87.3 |
| taiwan_late_payment | hellinger | 0.2938 | 0.6843 | 0.4299 | 0.2423 | 0.6011 | 142.929 | 3.927 | 255.0 |
| taiwan_late_payment | selected | 0.2925 | 0.6836 | 0.4323 | 0.2762 | 0.6142 | 50.620 | 1.480 | 87.3 |
| taiwan_late_payment | adaboost | 0.2836 | 0.6773 | 0.3999 | 0.2787 | 0.6151 | 692.909 | 6.673 | 51.7 |
| taiwan_late_payment | xgboost | 0.2936 | 0.6819 | 0.4345 | 0.2527 | 0.6057 | 68.903 | 6.279 | 1367.0 |
| taiwan_late_payment | lightgbm | 0.2919 | 0.6819 | 0.4341 | 0.2509 | 0.6051 | 86.164 | 15.404 | 1373.7 |
| rare_risk_20k | single | 0.0000 | 0.5000 | 0.0350 | 0.0000 | 0.5000 | 21.964 | 0.313 | 1.0 |
| rare_risk_20k | samme_30 | 0.6118 | 0.8706 | 0.4783 | 0.2667 | 0.6318 | 86.002 | 1.606 | 206.7 |
| rare_risk_20k | hellinger | 0.6860 | 0.8910 | 0.6078 | 0.3505 | 0.6743 | 216.141 | 6.288 | 719.7 |
| rare_risk_20k | selected | 0.6859 | 0.8874 | 0.6128 | 0.5181 | 0.7417 | 208.387 | 6.729 | 875.3 |
| rare_risk_20k | adaboost | 0.6815 | 0.8940 | 0.6395 | 0.4800 | 0.7383 | 10205.176 | 27.671 | 1250.0 |
| rare_risk_20k | xgboost | 0.7128 | 0.9013 | 0.6774 | 0.4705 | 0.7341 | 97.392 | 4.466 | 1348.0 |
| rare_risk_20k | lightgbm | 0.7120 | 0.9041 | 0.6837 | 0.4686 | 0.7335 | 119.768 | 18.413 | 1470.0 |

## Ablations

samme_30→samme_long changes only round count. samme_long→shrinkage changes only rate. shrinkage→hellinger changes only criterion; shrinkage→balanced changes only weighting. balanced→hellinger_balanced isolates the criterion under balancing. shrinkage→less_pruning changes only alpha. Comparisons to samme_30 combine changes.

| Task | Configuration | KS | AUC | AP | Rules | Accepted trees |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| german_credit | single | 0.0305 | 0.5152 | 0.3103 | 6.7 | 1.0 |
| german_credit | samme_30 | 0.4800 | 0.7750 | 0.5781 | 217.3 | 30.0 |
| german_credit | samme_long | 0.4673 | 0.7773 | 0.6058 | 767.7 | 100.0 |
| german_credit | shrinkage | 0.5016 | 0.7932 | 0.6015 | 754.0 | 100.0 |
| german_credit | balanced | 0.4895 | 0.7933 | 0.6051 | 809.0 | 100.0 |
| german_credit | hellinger | 0.4971 | 0.8009 | 0.6206 | 818.7 | 100.0 |
| german_credit | hellinger_balanced | 0.4806 | 0.8002 | 0.6418 | 839.7 | 100.0 |
| german_credit | gini | 0.4679 | 0.7991 | 0.6129 | 949.7 | 100.0 |
| german_credit | entropy | 0.4730 | 0.7916 | 0.6241 | 848.7 | 100.0 |
| german_credit | less_pruning | 0.4787 | 0.7833 | 0.5822 | 787.3 | 100.0 |
| credit_status | single | 0.3868 | 0.6934 | 0.4399 | 7.3 | 1.0 |
| credit_status | samme_30 | 0.5392 | 0.8267 | 0.6662 | 202.7 | 30.0 |
| credit_status | samme_long | 0.5368 | 0.8303 | 0.6765 | 606.3 | 100.0 |
| credit_status | shrinkage | 0.5430 | 0.8404 | 0.6919 | 594.3 | 100.0 |
| credit_status | balanced | 0.5426 | 0.8428 | 0.6908 | 599.3 | 100.0 |
| credit_status | hellinger | 0.5595 | 0.8436 | 0.6911 | 743.3 | 100.0 |
| credit_status | hellinger_balanced | 0.5582 | 0.8444 | 0.6906 | 731.7 | 100.0 |
| credit_status | gini | 0.5506 | 0.8416 | 0.6873 | 898.3 | 100.0 |
| credit_status | entropy | 0.5503 | 0.8408 | 0.6831 | 831.0 | 100.0 |
| credit_status | less_pruning | 0.5486 | 0.8400 | 0.6899 | 804.7 | 100.0 |
| taiwan_late_payment | single | 0.2301 | 0.6151 | 0.3389 | 2.0 | 1.0 |
| taiwan_late_payment | samme_30 | 0.2925 | 0.6836 | 0.4323 | 87.3 | 26.3 |
| taiwan_late_payment | samme_long | 0.2925 | 0.6836 | 0.4323 | 134.0 | 73.0 |
| taiwan_late_payment | shrinkage | 0.2938 | 0.6831 | 0.4288 | 213.0 | 100.0 |
| taiwan_late_payment | balanced | 0.2932 | 0.6817 | 0.4323 | 195.7 | 100.0 |
| taiwan_late_payment | hellinger | 0.2938 | 0.6843 | 0.4299 | 255.0 | 100.0 |
| taiwan_late_payment | hellinger_balanced | 0.2916 | 0.6849 | 0.4393 | 286.3 | 100.0 |
| taiwan_late_payment | gini | 0.2944 | 0.6844 | 0.4301 | 253.0 | 100.0 |
| taiwan_late_payment | entropy | 0.2942 | 0.6843 | 0.4298 | 245.3 | 100.0 |
| taiwan_late_payment | less_pruning | 0.2928 | 0.6828 | 0.4316 | 759.0 | 100.0 |
| rare_risk_20k | single | 0.0000 | 0.5000 | 0.0350 | 1.0 | 1.0 |
| rare_risk_20k | samme_30 | 0.6118 | 0.8706 | 0.4783 | 206.7 | 30.0 |
| rare_risk_20k | samme_long | 0.6403 | 0.8730 | 0.5289 | 708.7 | 100.0 |
| rare_risk_20k | shrinkage | 0.6423 | 0.8686 | 0.5062 | 535.0 | 100.0 |
| rare_risk_20k | balanced | 0.6473 | 0.8710 | 0.4782 | 647.3 | 100.0 |
| rare_risk_20k | hellinger | 0.6860 | 0.8910 | 0.6078 | 719.7 | 100.0 |
| rare_risk_20k | hellinger_balanced | 0.6771 | 0.8933 | 0.6050 | 793.3 | 100.0 |
| rare_risk_20k | gini | 0.6922 | 0.8867 | 0.6198 | 877.3 | 100.0 |
| rare_risk_20k | entropy | 0.7055 | 0.8940 | 0.6201 | 811.7 | 100.0 |
| rare_risk_20k | less_pruning | 0.6304 | 0.8648 | 0.5074 | 613.7 | 100.0 |

## Selection and cost

| Task | Seed | Training-CV choice | Search seconds |
| --- | ---: | --- | ---: |
| german_credit | 42 | gini | 1.601 |
| german_credit | 2024 | hellinger_balanced | 1.506 |
| german_credit | 7 | hellinger_balanced | 1.630 |
| credit_status | 42 | shrinkage | 2.687 |
| credit_status | 2024 | hellinger_balanced | 2.657 |
| credit_status | 7 | less_pruning | 2.710 |
| taiwan_late_payment | 42 | samme_30 | 3.053 |
| taiwan_late_payment | 2024 | samme_30 | 2.860 |
| taiwan_late_payment | 7 | samme_30 | 2.955 |
| rare_risk_20k | 42 | gini | 4.458 |
| rare_risk_20k | 2024 | hellinger_balanced | 4.853 |
| rare_risk_20k | 7 | gini | 4.863 |

## Scaling and variability

| Task | Interface | selected KS mean ± std | Fit 1-thread ms | Fit 4-thread ms | Fit speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| german_credit | python | 0.4629 ± 0.0270 | 56.676 | 55.496 | 1.02× |
| german_credit | cpp | 0.4629 ± 0.0270 | 64.393 | 56.329 | 1.14× |
| credit_status | python | 0.5541 ± 0.0065 | 129.510 | 87.686 | 1.48× |
| credit_status | cpp | 0.5541 ± 0.0065 | 127.014 | 91.144 | 1.39× |
| taiwan_late_payment | python | 0.2925 ± 0.0129 | 50.620 | 44.867 | 1.13× |
| taiwan_late_payment | cpp | 0.2925 ± 0.0129 | 50.434 | 49.649 | 1.02× |
| rare_risk_20k | python | 0.6859 ± 0.0137 | 208.387 | 169.687 | 1.23× |
| rare_risk_20k | cpp | 0.6859 ± 0.0137 | 217.187 | 173.807 | 1.25× |

All mean/std/min/max values for every engine/interface/thread configuration are available in the JSON. Small jobs run serially by design; timing fluctuations are not parallel gains.

## Findings

german_credit: baseline KS 0.4800 → train-CV selected 0.4629 (delta -0.0171); AUC 0.7750 → 0.7942. Criterion-only comparison: shrinkage KS 0.5016 → Hellinger 0.4971.

credit_status: baseline KS 0.5392 → train-CV selected 0.5541 (delta +0.0148); AUC 0.8267 → 0.8419. Criterion-only comparison: shrinkage KS 0.5430 → Hellinger 0.5595.

taiwan_late_payment: baseline KS 0.2925 → train-CV selected 0.2925 (delta +0.0000); AUC 0.6836 → 0.6836. Criterion-only comparison: shrinkage KS 0.2938 → Hellinger 0.2938.

rare_risk_20k: baseline KS 0.6118 → train-CV selected 0.6859 (delta +0.0741); AUC 0.8706 → 0.8874. Criterion-only comparison: shrinkage KS 0.6423 → Hellinger 0.6860.

Train-CV selection does not guarantee higher test KS on every task. German selected KS falls while AUC rises; Taiwan keeps baseline; credit status and rare risk gain KS. Defaults were not changed based on test outcomes. More rounds are not always better. The criterion-only Hellinger ablation improves credit status/rare risk KS, but slightly reduces German KS. Balancing improves bad recall without guaranteeing KS/AP gains.

Three repetitions add timing evidence, not independent accuracy evidence; the three splits overlap. No statistical-significance or universal-superiority claim is made. XGBoost/LightGBM retain higher KS on rare risk. selected has extra search budget. Rule counts are structural proxies, not a human comprehension study.

## Validation

New tests failed before implementation. Hellinger binary behavior/multiclass rejection, explicit off/one-tree equivalence, balanced/user-weight equivalence, failed-refit preservation, tie-safe/directional KS, per-fold encoders, CV isolation and cache integrity pass. Release CTest 7/7; Python estimator/inspection 70/70; CLI 16/16; performance/RSS 3/3; benchmark methodology 15/15. ASan/UBSan/LSan CTest 7/7 plus CLI 16/16; TSan 7/7. Native lines 2444/2444 and Python statements 371/371: 100%. No new coverage exclusions. Benchmark code is outside production coverage gates.

## Reproduce

```sh
python -m pip install '.[dev,benchmark]'
make test
make coverage
make sanitize
make thread-sanitize
PYTHONPATH=python:. python benchmarks/risk.py --seeds 42 2024 7 --threads 1 4 \
  --rounds 100 --folds 3 --repeats 3 --output benchmarks/risk-results.json
PYTHONPATH=python:. python benchmarks/risk_report.py
```


Measured source SHA-256: `7e1d4b7b56c059749876fbb0e873b0b0e6c39bfa92b5619ae7d9007ebe190c4c`. Source URLs/checksums, all per-fold CV scores, chosen parameters and raw repeated measurements are in the JSON. Downloaded data and prepared predictions remain ignored by git.
