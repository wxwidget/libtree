# Improving C5-style models for risk scoring

[中文](c5-risk.zh-CN.md) · [risk benchmark and results](../benchmarks/RISK_REPORT.md) ·
[core C5 guide](c5.md)

The existing classifier already uses discrete multiclass SAMME. This extension
adds an explicit boosting switch, class balancing, a binary Hellinger criterion,
and a reproducible train-only model-selection experiment. No default criterion
is changed after looking at the benchmark test outcomes.

## Implemented techniques and attribution

| Technique | Implementation | Why evaluate it? | Reference |
| --- | --- | --- | --- |
| Discrete boosting | `boosting='samme'`, `n_estimators`, `learning_rate` | More rounds can improve ranking; shrinkage slows weight concentration | Zhu et al. [1] |
| Single-tree control | `boosting='none'` | Separates boosting gains from tree quality; keeps one tree regardless of the requested round budget | [1] |
| Inverse-frequency weights | `class_weight='balanced'` | Penalizes minority errors; multiplied with user sample weights | He & Garcia [2] |
| Hellinger splitting | `criterion='hellinger'` | Uses class-conditional branch distributions rather than class priors | Cieslak et al. [3] |
| Train-only selection | Stratified CV with a minimum directional-KS improvement | Avoids selecting a method on the final test set | Cawley & Talbot [4] |
| Cost-complexity control | Existing alpha, plus a weaker-pruning ablation | Checks the quality/readability tradeoff | Breiman et al.; [core references](c5.md#references-and-attribution) |

References are also placed next to the weight calculation in
`python/libtree/c5.py`, the SAMME/Hellinger calculations in `src/c5.cc`, and
selection/KS calculations in the benchmark code. All implementation code is
independent; references do not imply full compatibility with those systems.

For a binary split, let `L0/T0` and `L1/T1` be the fractions of each class
routed left, and `R0/T0`, `R1/T1` the right fractions. Hellinger score is:

```text
sqrt((sqrt(L0/T0)-sqrt(L1/T1))² + (sqrt(R0/T0)-sqrt(R1/T1))²)
```

It lies in [0, sqrt(2)]. Current SAMME/sample weights enter all counts. This
score removes the direct class-prior factor; the full classifier is still
affected by weighted leaf decisions, pruning, global quantiles and boosting.
`min_gain` has Hellinger-distance units for this criterion. It is binary-only
(degenerate one-class fits are allowed); multiclass fits fail without replacing
the old model. Missing routing is learned as before. No logarithms are needed
in the Hellinger scoring path.

Balanced weights are `N / (K * class_count)`, computed from the fitting labels.
With supplied sample weights, multiply the two weight vectors, then the native
learner normalizes them. Inputs are not mutated. More minority recall can come
with lower overall accuracy or more false positives; evaluate both classes.
Balancing does not calibrate probabilities or encode application-specific costs.

## Use the parameters

```python
import numpy as np
from libtree import C5Classifier
from sklearn.model_selection import train_test_split
from benchmarks.risk import dataset, encode  # Run from the repository checkout.

frame, labels, source = dataset('german_credit')
frame = frame[['duration', 'amount', 'age']]  # Numeric teaching subset; benchmark uses all eligible predictors.
names = sorted(frame.columns)  # encode orders numeric columns alphabetically.
a, b = train_test_split(np.arange(len(labels)), test_size=.25,
                        stratify=labels, random_state=42)
X_train, X_test = encode(frame.iloc[a], frame.iloc[b])
y_train = labels[a]

model = C5Classifier(boosting='samme', n_estimators=100, learning_rate=.3,
                     criterion='hellinger', class_weight='balanced',
                     max_depth=4, pruning_alpha=.001, n_jobs=4)
# X_train and y_train come from the training partition; y=1 means bad/risk.
model.fit(X_train, y_train)
scores = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
print(model.explain(X_test[0], feature_names=names, format='text'))
model.close()
```

This is an example candidate, not a recommended universal configuration. Use
validation results to choose it. `boosting='none'` fits exactly one tree; the
requested `n_estimators` must still be valid. `get_params`/`set_params` include
both new settings; changes apply at the next fit. Default behavior remains
SAMME, rate 1 and no class balancing. Pruning and exported-rule semantics are
unchanged. Predictive probabilities remain uncalibrated normalized-vote softmax.

```sh
PYTHONPATH=python python -m libtree rules --train credit.csv --header \
  --boosting samme --rounds 100 --rate .3 --criterion hellinger \
  --class-weight balanced --depth 4 --alpha .001 --threads 4
PYTHONPATH=python python -m libtree rules --train credit.csv --header --boosting none
```

C++ uses `C5Parameters::num_estimators`, `learning_rate` and
`SplitCriterion::kHellinger`. Set `num_estimators=1` for the single-tree control.
Pass explicit inverse-frequency weights to `Fit` for balancing. Existing C ABI
criterion IDs 0/1/2 remain unchanged; 3 selects Hellinger. The native benchmark
supports the same weighted fit and checks Python/native probability parity.

## Risk metrics and experiment design

The benchmark consistently defines label 1 as the bad/risk event and higher
scores as more risk. KS is the maximum absolute difference between the empirical
bad/good score distributions [5]. We also expose directional KS,
`max(TPR-FPR)`, useful for detecting inverted scores. Scores that are equal form
one threshold group. A constant score gives KS=0; reversed perfect scores have
absolute KS=1 but directional KS=0 and AUC=0. Always read KS together with AUC.

Test-set KS is a descriptive ranking statistic, not a deployed threshold selected
on that test set. Binary KS is added to both the original C5 benchmark metrics
and the risk-specific study. Multiclass KS is undefined and omitted. The risk
study also records average precision, bad-class recall, balanced accuracy,
accuracy, log loss, train accuracy, rules, conditions and train/predict times.

Three real tasks are German credit risk, credit good/bad status, and Taiwan
late payment. The Taiwan public subset has April predictors and a September
late-payment outcome; it is not the full original credit-default dataset.
Identifiers/targets are removed, data source SHA-256 is pinned. A separate
20k-row task is explicitly synthetic, with rare risk events and label noise.
Random stratified holdouts are not temporal backtests, policy simulations or
proof that the models work on another lender's population.

For each of three outer 75/25 splits, ten fixed C5 configurations undergo
three-fold stratified CV on the outer training partition. Category vocabularies
are refitted in each inner fold. Rank candidates by mean directional KS, then
AUC, then fewer rules. Require an absolute KS gain of .01 over `samme_30`,
otherwise keep the baseline. Refit the chosen configuration on all outer
training rows, then evaluate the untouched outer test set. This gate is a
selection policy, not a guarantee that test performance improves.

The candidate set separately changes round budget, rate, balancing, Hellinger,
Gini/entropy and pruning. XGBoost, LightGBM and sklearn SAMME receive fixed
100-round budgets. They are not separately tuned; the selected C5 result has an
extra search budget, whose cost and all fold scores are disclosed. Raw C5
ablations remain available for direct fixed-configuration comparisons.

## Next techniques to evaluate

1. **Repeated/nested CV and stability:** especially for small credit datasets,
   quantify how sensitive selected rules are to the training split [4].
2. **Real/probabilistic boosting:** class-distribution leaf scores may improve
   smooth ranking [1], but require new faithful rule-contribution semantics;
   do not present this as implemented SAMME.R.
3. **Calibration on separate validation data:** sigmoid/isotonic methods may
   improve probability quality [6]. Monotone calibration usually preserves
   ranking and cannot be advertised as a KS/AUC gain.
4. **Cost-aware operating thresholds:** select against false-positive/negative
   costs on validation data [2], then freeze before test/deployment; this can
   improve an operating decision without improving the whole ranking.
5. **More representative evaluation:** full credit/default features, time-based
   folds, population drift and external cohorts before changing defaults.

Keep baseline configurations, source hashes and raw results. Retain an
optimization only when fresh validation/test evidence supports its particular
quality, speed or readability benefit.

## References

1. Zhu, Zou, Rosset, Hastie (2009), “Multi-class AdaBoost,”
   [doi:10.4310/SII.2009.v2.n3.a8](https://doi.org/10.4310/SII.2009.v2.n3.a8).
2. He & Garcia (2009), “Learning from Imbalanced Data,” IEEE TKDE,
   [doi:10.1109/TKDE.2008.239](https://doi.org/10.1109/TKDE.2008.239).
3. Cieslak, Ho, Chawla, Kegelmeyer (2012), “Hellinger distance decision trees
   are robust and skew-insensitive,” DMKD 24, 136–158,
   [doi:10.1007/s10618-011-0222-1](https://doi.org/10.1007/s10618-011-0222-1).
4. Cawley & Talbot (2010), “On Over-fitting in Model Selection and Subsequent
   Selection Bias in Performance Evaluation,” JMLR 11, 2079–2107,
   [paper](https://jmlr.org/papers/v11/cawley10a.html).
5. Massey (1951), “The Kolmogorov-Smirnov Test for Goodness of Fit,” JASA,
   [doi:10.1080/01621459.1951.10500769](https://doi.org/10.1080/01621459.1951.10500769).
6. Niculescu-Mizil & Caruana (2005), “Predicting Good Probabilities with
   Supervised Learning,” ICML,
   [doi:10.1145/1102351.1102430](https://doi.org/10.1145/1102351.1102430).

Risk sources: [German mirror](https://github.com/selva86/datasets/blob/master/GermanCredit.csv),
[credit status documentation](https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/credit_data.html),
[Taiwan subset documentation](https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/credit_cards.html).
