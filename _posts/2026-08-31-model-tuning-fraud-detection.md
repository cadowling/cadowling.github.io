---
layout: post
title: Model Tuning and Optimization with Fraud Detection
image: "/posts/fraud-detection-tuning-title-img.png"
tags: [Machine Learning, XGBoost, Hyperparameter Tuning, Bayesian Optimization, Python]
---

In this project I build a credit card fraud detection model, and then put it through **three different tuning strategies** to understand what each one actually buys me.

I begin by establishing baseline models on a severely imbalanced dataset, where only **0.167%** of transactions are fraudulent. I then tune the model's hyperparameters two ways — exhaustively with **grid search**, and adaptively with **Bayesian optimization** using Optuna — before finally tuning the **decision threshold** on top of both tuned models.

The insight running through the whole project is that these techniques are not competitors. Two of them tune the *model*; one of them tunes the *decision rule*. Understanding that difference is what turns a well-scoring classifier into a genuinely useful one.

# Table of Contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. Why Imbalance Breaks Standard Metrics](#imbalance-overview)
- [03. Data Preparation](#data-prep)
- [04. Baseline Models](#baseline-models)
- [05. Tuning With Grid Search](#tuning-grid)
- [06. Tuning With Bayesian Optimization](#tuning-bayesian)
- [07. Tuning With Thresholding](#tuning-threshold)
    - [Choosing the Threshold Without Leaking](#threshold-leak)
    - [Three Threshold Strategies](#threshold-strategies)
    - [Applying Thresholding to the Tuned Models](#threshold-applied)
- [08. Results Comparison](#results-comparison)
- [09. Growth & Next Steps](#growth-next-steps)

___

# 00. Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Credit card fraud detection is a classic **extreme class imbalance** problem. In this dataset, fewer than 2 transactions in every 1,000 are fraudulent.

This creates a trap that catches a lot of first-pass models: a classifier that simply predicts "not fraud" for every single transaction achieves **99.83% accuracy** while catching precisely zero fraud. Any tuning process that optimises the wrong metric will happily walk straight into that trap.

The business problem is also not symmetric. A missed fraud costs the money lost on the transaction. A false alarm costs an analyst's review time and annoys a legitimate customer. These are different currencies, and the right balance between them is a business decision, not a statistical one.

### Actions <a name="overview-actions"></a>

I built and compared a full tuning pipeline that:

* Cleaned the data and removed duplicate records
* Scaled the two non-anonymised features using `RobustScaler`
* Established baselines with Random Forest, XGBoost and LightGBM
* Selected **PR-AUC (average precision)** as the optimisation target instead of accuracy or ROC-AUC
* Tuned hyperparameters **exhaustively** with `GridSearchCV`
* Tuned hyperparameters **adaptively** with Optuna's TPE sampler and median pruning
* Tuned the **decision threshold** on top of both tuned models, using out-of-fold predictions and three different business strategies

Throughout, I used stratified cross-validation so that every fold retained the same fraud rate.

### Results <a name="overview-results"></a>

The headline finding is that **hyperparameter search and thresholding do completely different jobs**:

* Hyperparameter search changes how well the model *ranks* transactions, which is what PR-AUC measures.
* Thresholding never changes PR-AUC at all. It only chooses where to cut that ranking — trading precision against recall along a fixed curve.
* On the baseline model, simply moving the cut-off from 0.5 to 0.7084 raised precision from 0.936 to 0.973 with no loss of recall at all.
* Hitting a 90% recall target was possible but expensive: false alarms rose from 5 to 1,733.

### Growth/Next Steps <a name="overview-growth"></a>

Potential future enhancements include:

* Cost-sensitive thresholding using real currency values rather than F1
* Time-based validation splits to respect the temporal ordering of transactions
* Calibration analysis, since LightGBM's baseline showed severe miscalibration
* Ensembling the tuned XGBoost and LightGBM models
* Monitoring for concept drift as fraud patterns evolve

___

# 01. Data Overview <a name="data-overview"></a>

The dataset contains **284,807 credit card transactions**, of which a tiny fraction are labelled as fraudulent.

Most of the features are the output of a PCA transformation applied for confidentiality, so they arrive as anonymous components `V1` through `V28`. Only three columns are human-readable:

| Column | Meaning |
|---|---|
| `Time` | Seconds elapsed since the first transaction in the dataset |
| `Amount` | Transaction amount |
| `V1`–`V28` | Anonymised principal components |
| `Class` | Target: 1 = fraud, 0 = normal |

After removing **1,081 duplicate records**, I was left with 283,726 transactions, distributed as follows:

```
--- Class Distribution ---
Normal (0): 283253 transactions (99.833%)
Fraud (1): 473 transactions (0.167%)
```

<br>
**Why this matters:**  With only 473 positive examples in the entire dataset, every modelling decision from this point forward has to be made with the imbalance in mind. There is very little fraud to learn from, and a great deal of normal behaviour to be distracted by.

___

# 02. Why Imbalance Breaks Standard Metrics <a name="imbalance-overview"></a>

Before tuning anything, I needed to be sure I was tuning *toward* the right thing. Two commonly used metrics are actively misleading here.

**Accuracy is useless.** Predicting "not fraud" every time scores 99.83%. Any optimiser handed accuracy as an objective will find that solution and stop.

**ROC-AUC is optimistically biased.** ROC-AUC plots true positive rate against false positive rate. The false positive rate has the size of the negative class in its denominator, and that class has 283,253 members. Flagging 1,000 legitimate transactions moves the false positive rate by roughly 0.35% — barely visible on a ROC curve, but a catastrophe for the analyst who has to review all 1,000.

**PR-AUC is the honest choice.** Precision-recall curves ignore true negatives entirely, so a model gets no credit for the easy majority class. This is why every search in this project optimises `average_precision`.

The gap is visible in my own results: the baseline XGBoost model scored **0.9736 ROC-AUC** but only **0.8224 PR-AUC**. The first number flatters the model; the second describes it.

<br>
**Why this matters:**  Choosing the metric is the single highest-leverage decision in an imbalanced problem. Every tuning method below is only as good as the objective it is pointed at.

___

# 03. Data Preparation <a name="data-prep"></a>

The PCA components are already centred and scaled. `Time` and `Amount` are not, so I scaled them with `RobustScaler`, which uses the median and interquartile range rather than the mean and standard deviation.

```python
from sklearn.preprocessing import RobustScaler

rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)
```

<br>
**Why this matters:**  Transaction amounts are heavily right-skewed with extreme outliers. `StandardScaler` would let a handful of very large transactions dominate the mean and variance; `RobustScaler` is far less sensitive to them.

I then split into train and test sets, **stratified on the target** so that both sets retain the same fraud rate:

```python
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```

This gives 226,980 training transactions containing 378 frauds, and 56,746 test transactions containing 95 frauds.

<br>
**Why this matters:**  Without `stratify=y`, random chance alone could hand me a test set with a materially different fraud rate, making every downstream comparison unreliable.

___

# 04. Baseline Models <a name="baseline-models"></a>

I established three baselines before tuning anything. For the gradient boosting models I set `scale_pos_weight` to the ratio of negative to positive cases (**599.48**), which is the boosting equivalent of `class_weight='balanced'`.

```python
neg_class_count = (y_train == 0).sum()
pos_class_count = (y_train == 1).sum()
scale_weight = neg_class_count / pos_class_count
```

At the default 0.5 decision threshold, the three baselines performed as follows on the fraud class:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Random Forest | 0.79 | 0.76 | 0.77 |
| XGBoost | 0.83 | 0.81 | 0.82 |
| LightGBM | 0.02 | 0.87 | 0.05 |

The LightGBM result is worth pausing on. It has the **best recall of the three** and a catastrophic precision of 0.02, meaning roughly 98 out of every 100 transactions it flagged were legitimate. Given the identical `scale_pos_weight`, this is not a difference in predictive power so much as a difference in where each model places its probabilities — which is precisely the problem that thresholding exists to solve later on.

I carried XGBoost forward as the model to tune.

___

# 05. Tuning With Grid Search <a name="tuning-grid"></a>

Grid search fits **every combination** in the grid. That makes it exhaustive, perfectly reproducible and trivially parallel, but the cost is the product of the list lengths — adding one more three-value parameter triples the runtime.

I kept the grid deliberately small and centred on the baseline values, pinning the remaining knobs so the search stayed tractable:

```python
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.3],
    'n_estimators': [100, 300],
    'subsample': [0.8, 1.0],
}

grid_search = GridSearchCV(
    estimator=xgb_grid_base,
    param_grid=param_grid,
    scoring='average_precision',   # PR-AUC, not accuracy
    cv=cv,
    n_jobs=1,                      # XGBoost already uses every core
    verbose=1,
    refit=True)

grid_search.fit(X_train, y_train)
```

That is 36 combinations across 3 folds, or **108 model fits**, to explore just four parameters.

<br>
**Why this matters:**  Setting `n_jobs=-1` on both the estimator and the search oversubscribes the CPU and runs *slower*, not faster. Parallelise in one place only.

I also inspected the full CV results table rather than just the winner, because it usually reveals a broad plateau of near-identical scores — which is exactly why exhaustively sweeping a lattice is such a poor use of compute.

### Results

*[To be completed from the notebook run.]*

| Metric | Value |
|---|---|
| Runtime | |
| Model fits | 108 |
| Best CV PR-AUC | |
| Best parameters | |
| Score spread across grid | |
| Test PR-AUC | |

___

# 06. Tuning With Bayesian Optimization <a name="tuning-bayesian"></a>

Bayesian optimization builds a probabilistic model of *(hyperparameters → score)* from the trials run so far, and uses it to choose each next trial. Instead of sweeping a fixed lattice, it concentrates the budget on promising regions.

This has two practical consequences that matter a great deal here:

1. Parameters are drawn from **continuous ranges**, so the search is not restricted to the handful of values I happened to type into a grid.
2. The budget is set by `n_trials`, **independently of how many parameters are being tuned**. Adding a tenth parameter costs no extra runtime.

That second point is why grid search is the wrong tool for XGBoost specifically. XGBoost has around nine knobs worth tuning, and that is exactly where a grid goes exponential. Here I tune all nine on a fixed budget of 40 trials.

```python
def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, scale_weight, log=True),
        'random_state': 42,
        'n_jobs': -1,
    }

    # Manual fold loop so intermediate scores can be reported to the pruner
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train)):
        model = xgb.XGBClassifier(**params)
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        va_proba = model.predict_proba(X_train.iloc[va_idx])[:, 1]
        fold_scores.append(average_precision_score(y_train.iloc[va_idx], va_proba))

        trial.report(float(np.mean(fold_scores)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))

study.optimize(objective, n_trials=40, show_progress_bar=True)
```

Two design choices are worth calling out.

**`scale_pos_weight` is treated as tunable**, not pinned at the class ratio. The full ratio of 599 is the right choice for balanced accuracy, but it tends to over-predict fraud when the objective is PR-AUC. I let the search decide.

**Pruning abandons unpromising trials mid-cross-validation.** Rather than using `cross_val_score`, I loop the folds manually so an intermediate score can be reported after each one. This is where most of the saving over grid search actually comes from.

Optuna also reports **which parameters mattered**, via fANOVA importance — something grid search cannot do:

```python
importances = optuna.importance.get_param_importances(study)
```

### Results

*[To be completed from the notebook run.]*

| Metric | Value |
|---|---|
| Runtime | |
| Trials / pruned | 40 / |
| Best CV PR-AUC | |
| Best parameters | |
| Most important parameters | |
| Test PR-AUC | |

___

# 07. Tuning With Thresholding <a name="tuning-threshold"></a>

Both searches above produce a model that *ranks* transactions by fraud probability. Neither of them decides where to draw the line. That decision is the threshold, and it is **not a hyperparameter** — it is a cut-off applied to predicted probabilities after the model is trained.

This makes it the cheapest tuning step in the whole project. It costs a single model fit rather than hundreds, and at a 0.167% positive rate the default 0.5 is almost never the operating point I actually want.

## Choosing the Threshold Without Leaking <a name="threshold-leak"></a>

This is the step that is most often got wrong. Sweeping thresholds on the test set and reporting the best one leaks the test set into the model and inflates the result.

Instead I use `cross_val_predict` to produce an **out-of-fold probability for every training row**. Each prediction comes from a model that never saw that row, so the whole training set can be used to choose the threshold honestly.

```python
from sklearn.model_selection import cross_val_predict, StratifiedKFold

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

oof_proba = cross_val_predict(model, X_train, y_train, cv=cv,
                              method='predict_proba', n_jobs=1)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_train, oof_proba)
# precision_recall_curve returns one more precision/recall point than thresholds,
# so trim the last element off both
precision, recall = precision[:-1], recall[:-1]
```

<br>
**Why this matters:**  A threshold tuned on the test set is not a threshold at all, it is a leak. Out-of-fold predictions cost three model fits and make the number trustworthy.

## Three Threshold Strategies <a name="threshold-strategies"></a>

There is no single "correct" threshold, because the right answer depends on what the business is optimising. I implemented three.

**Strategy 1 — Maximise F1.** The balanced default, appropriate when a missed fraud and a false alarm are treated as roughly equally costly.

```python
f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
t_f1 = thresholds[np.argmax(f1_scores)]
```

**Strategy 2 — Fix recall, maximise precision.** Much closer to how fraud teams actually operate: *"we must catch 90% of fraud; minimise how many good customers we annoy."* I take the highest threshold that still clears the recall floor.

```python
TARGET_RECALL = 0.90
meets_recall = np.where(recall >= TARGET_RECALL)[0]
t_recall = thresholds[meets_recall[-1]]
```

**Strategy 3 — Cap the alert volume.** When a review team can only work N cases a day, the threshold is set by capacity, not by any statistical criterion.

```python
ALERT_BUDGET = 0.001   # flag at most 0.1% of transactions
t_budget = np.quantile(oof_proba, 1 - ALERT_BUDGET)
```

## Applying Thresholding to the Tuned Models <a name="threshold-applied"></a>

Because thresholding is independent of the search that produced the model, the same procedure can be applied to both tuned models. I run the out-of-fold sweep once per model:

```python
from sklearn.base import clone

tuned_models = {'GridSearchCV': xgb_grid, 'Optuna': xgb_optuna}

for name, fitted_model in tuned_models.items():
    oof = cross_val_predict(clone(fitted_model), X_train, y_train, cv=cv,
                            method='predict_proba', n_jobs=1)[:, 1]

    p, r, t = precision_recall_curve(y_train, oof)
    p, r = p[:-1], r[:-1]
    f1 = 2 * p * r / (p + r + 1e-12)
    best_t = t[np.argmax(f1)]

    print(f"{name}: tuned threshold = {best_t:.4f}")
    evaluate(f"XGBoost - {name} + tuned threshold",
             fitted_model, X_test, y_test, best_t)
```

### Thresholding the Grid Search Model

*[To be completed from the notebook run.]*

| Strategy | Threshold | PR-AUC | Precision | Recall | Frauds caught | False alarms |
|---|---|---|---|---|---|---|
| Default | 0.5000 | | | | | |
| Best F1 | | | | | | |
| Recall ≥ 90% | | | | | | |
| 0.1% alert budget | | | | | | |

### Thresholding the Bayesian Optimized Model

*[To be completed from the notebook run.]*

| Strategy | Threshold | PR-AUC | Precision | Recall | Frauds caught | False alarms |
|---|---|---|---|---|---|---|
| Default | 0.5000 | | | | | |
| Best F1 | | | | | | |
| Recall ≥ 90% | | | | | | |
| 0.1% alert budget | | | | | | |

### What Thresholding Actually Does

To illustrate the mechanism before the tuned numbers are filled in, here is the same sweep applied to the **untuned baseline XGBoost model**. All four rows come from one fitted model with identical predicted probabilities — only the cut-off changes.

| Strategy | Threshold | PR-AUC | Precision | Recall | Frauds caught | False alarms |
|---|---|---|---|---|---|---|
| Default | 0.5000 | 0.8224 | 0.9359 | 0.7684 | 73 / 95 | 5 |
| Best F1 | 0.7084 | 0.8224 | 0.9733 | 0.7684 | 73 / 95 | 2 |
| Recall ≥ 90% | 0.0002 | 0.8224 | 0.0468 | 0.8947 | 85 / 95 | 1,733 |
| 0.1% alert budget | 0.9999 | 0.8224 | 0.9825 | 0.5895 | 56 / 95 | 1 |

Three things stand out, and all three should hold for the tuned models too.

**PR-AUC is identical in every row.** This is the clearest possible demonstration that thresholding does not improve the model. It cannot change how well the model *ranks* transactions; it only chooses where to cut that ranking. Any PR-AUC improvement in this project has to come from the hyperparameter search, never from the threshold.

**Raising the threshold to 0.7084 was free money.** Recall was unchanged at 73 frauds caught, but false alarms dropped from 5 to 2 and precision rose from 0.936 to 0.973. The default 0.5 was simply the wrong cut-off.

**The 90% recall target was brutally expensive.** Catching 12 additional frauds required dropping the threshold to 0.0002 and accepting **1,733 false alarms**, collapsing precision to 0.047. This is exactly the kind of trade-off that should be surfaced to a business stakeholder rather than buried in a metric.

<br>
**A note on the alert budget:**  the 0.1% budget caps recall on arithmetic alone — 0.1% of 56,746 transactions is only 57 alerts, and there are 95 frauds to find, so even a perfect model could not exceed 60% recall. Setting the budget above the 0.167% fraud base rate would make this strategy meaningfully competitive.

___

# 08. Results Comparison <a name="results-comparison"></a>

*[Tuned rows to be completed from the notebook run.]*

| Approach | Threshold | PR-AUC | Precision | Recall | Frauds missed | False alarms |
|---|---|---|---|---|---|---|
| XGBoost — baseline, default threshold | 0.5000 | 0.8224 | 0.9359 | 0.7684 | 22 | 5 |
| XGBoost — baseline, best-F1 threshold | 0.7084 | 0.8224 | 0.9733 | 0.7684 | 22 | 2 |
| XGBoost — GridSearchCV | 0.5000 | | | | | |
| XGBoost — GridSearchCV + tuned threshold | | | | | | |
| XGBoost — Optuna | 0.5000 | | | | | |
| XGBoost — Optuna + tuned threshold | | | | | | |

The way to read this table is to remember that **PR-AUC only moves when the model changes, and precision/recall only move when the threshold changes.** That single rule makes it obvious which lever each method is pulling.

### Summary of the three approaches

| | What it tunes | Cost | Strengths | Limitations |
|---|---|---|---|---|
| **Grid search** | Hyperparameters | ∏(list lengths) × folds | Exhaustive, reproducible, trivially parallel, no extra dependency | Cost is multiplicative; wastes most fits on a plateau; restricted to the values you typed |
| **Bayesian (Optuna)** | Hyperparameters | Fixed `n_trials` | Budget independent of parameter count; continuous ranges; pruning; parameter importance | Sequential, so parallelises less well; stochastic; extra dependency |
| **Thresholding** | The decision cut-off, post-fit | 1 fit | Nearly free; directly expresses business cost; highest-leverage knob under extreme imbalance | Cannot improve the ranking — PR-AUC is unchanged |

The practical conclusion is that these are **complementary, not competing**, and the ordering in this project reflects that. Search the hyperparameters first to get the best possible ranking of transactions, then tune the threshold on that model to choose the operating point the business actually wants. Doing it the other way round means re-tuning the threshold every time the model changes.

___

# 09. Growth & Next Steps <a name="growth-next-steps"></a>

Potential future enhancements include:

* **Cost-sensitive thresholding.** Replace F1 with a real currency objective — expected fraud loss avoided minus review cost — so the threshold is chosen on money rather than on a symmetric statistic.
* **Time-aware validation.** Transactions are ordered in time, and a random split lets the model learn from the future. A rolling time-based split would give a more honest estimate.
* **Calibration analysis.** LightGBM's 0.02 baseline precision suggests severe miscalibration; Platt scaling or isotonic regression would make its probabilities directly comparable to XGBoost's.
* **Ensembling.** The tuned XGBoost and LightGBM models make different errors, and averaging their probabilities is a cheap way to improve ranking.
* **Drift monitoring.** Fraud patterns change adversarially, so the threshold and the model both need periodic revalidation against recent data.

This project demonstrates that on a severely imbalanced problem, choosing the right *metric* and the right *operating point* can matter as much as the hyperparameter search itself.

___
