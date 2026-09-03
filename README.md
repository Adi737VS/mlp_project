# Heavy Equipment Selling Price Prediction

Regression on used heavy-machinery auction records: predict the sale price of a machine from
its specification, usage history and sale date.

Built for a Kaggle competition as part of the **IIT Madras BS in Data Science** programme
(Machine Learning Practice).

| | |
|---|---|
| **Task** | Regression — predict `TargetValue` (USD sale price) |
| **Metric** | RMSLE (root mean squared logarithmic error) |
| **Data** | 138,701 train × 54 features after preprocessing, 15,000 test |
| **Best single model** | LightGBM — 0.20076 validation RMSLE |
| **Final submission** | 0.8 × LightGBM + 0.2 × XGBoost |

---

## Approach

The dataset is categorical-heavy, missing-data-heavy, and has a strongly skewed target. Twelve
EDA investigations were run first, each one motivating a specific downstream decision rather
than serving as decoration.

### Four findings that shaped the model

**1. The target is right-skewed — train in log space.**
Raw prices have skew 0.970; `log1p` brings this to −0.198. The deeper reason for the transform
is that **RMSLE is RMSE measured in log space**, so training on the log target with ordinary
squared error optimises the competition metric directly instead of approximating it.

**2. Missingness is structured, not random.**
Correlating per-column missingness indicators revealed six distinct blocks of columns sharing
identical missing rates (48.8%, 78.9%, 83.1%, 86.4%, 89.2%, 92.5%). Columns that vanish
together point to a record source that only populates certain fields — meaning the missingness
pattern itself carries information about a row's provenance.

**3. Operating hours vs price is a Simpson's paradox.**
Pooled across all rows, more operating hours appears to correlate with a *higher* price. Within
a single machine class, the correlation flips negative — 82% of classes show a negative
relationship (IQR −0.459 to −0.076).

The confounder is machine size: large machines are both expensive and heavily used, and pooling
lets the size effect swamp the usage effect. A linear model would learn the wrong sign on a key
feature. Tree models can condition on machine class before splitting on hours, which recovers
the correct relationship — a concrete, data-driven argument for the model family chosen.

**4. Machine identity is highly predictive.**
120,111 unique machines across 138,701 rows, with 31,447 repeat-sale rows. Once `AssetID` is
known, only **4.1%** of log-price variance remains. 23.3% of test rows involve a machine already
priced in the training set, so `AssetID` is retained as a feature.

### Additional EDA

| Analysis | Finding |
|---|---|
| Volume and price by year | Test share constant at 9.7% ± 0.87pp per year → **random split, not chronological**, validating a random train/validation split |
| Price vs machine age | Correlation −0.373; mean log price falls 10.93 → 9.47 over 40 years |
| Categorical cardinality | Up to 3,594 levels; unseen-in-train levels in up to 0.59% of test rows |
| Target granularity | Only 675 distinct prices, bounded 7,500–142,000 — effectively discrete and clipped |
| Date feature signal | Month spread 13.6%, weekday spread 18.0%, with 67.8% of sales Wed–Fri |

---

## Preprocessing

**Feature engineering**
- Date decomposed into year, month, weekday, day-of-year, plus `sl_days` — a monotonic index of
  days since 1990-01-01, letting a single tree split separate any before/after date boundary
- `age` = sale year − manufacture year, clipped to 0–60
- `log_hrs` = `log1p(OperationalHoursMeter)`

**Cleaning**
- Placeholder values converted to genuine `NaN`: manufacture years below 1900 (encoded as 0 or
  1000) and hour-meter readings of exactly 0. Without this guard, a machine with year 1000
  would receive `age = 1005`, which `.clip(0, 60)` would silently reduce to 60 —
  indistinguishable from a genuine 60-year-old machine.
- `ProductConfigID` cast to string so it is treated as a category rather than a quantity;
  as an integer, a model could split on `≤ 1800`, which performs arithmetic on an identifier.

**Encoding and imputation**

| Column type | Treatment | Reason |
|---|---|---|
| Numeric | `SimpleImputer(strategy="median")` | Skewed distributions; median is outlier-robust |
| Categorical | `SimpleImputer(strategy="most_frequent")` → `OrdinalEncoder` | Cardinality up to 3,594 makes one-hot infeasible (~8,700 near-empty columns) |

`OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` handles categories
present in test but absent from train, directly addressing what the cardinality analysis
measured.

No feature scaling is applied — tree models split on thresholds and are invariant to monotonic
rescaling.

**Leakage control.** The `ColumnTransformer` is fitted with `fit_transform` on training rows
only, then applied with `transform` to validation and test. Imputation statistics and category
mappings never see held-out data.

---

## Models

Five families compared on an identical 80/20 split (`random_state=42`), each tuned with
`RandomizedSearchCV` and refitted with the selected configuration.

| Model | Train RMSLE | Validation RMSLE | Gap | Fit time |
|---|---|---|---|---|
| Decision Tree | 0.24115 | 0.29130 | 0.050 | 2.3 s |
| HistGradientBoosting | 0.17363 | 0.21549 | 0.042 | 26.9 s |
| **LightGBM** | 0.08337 | **0.20076** | 0.117 | 50.3 s |
| CatBoost | 0.20952 | 0.29160 | 0.082 | 168.2 s |
| XGBoost | 0.15328 | 0.20643 | 0.053 | 29.4 s |

The single decision tree serves as a reference point: it is the simplest model capturing
non-linearity and interactions, so the margin between it and the ensembles measures what
boosting contributes.

**On the train/validation gap.** LightGBM has by far the widest gap yet the best validation
score. The gap measures how hard a model fits its training data, not how well it generalises —
`num_leaves=350` with `min_child_samples=8` is a high-capacity configuration, counterbalanced by
L1/L2 penalties and column/row subsampling. It nevertheless generalises best of the five.

---

## Ensemble

The final submission blends the two strongest models in log space before inverting:

```python
log_pred = 0.8 * lgbm_pred + 0.2 * xgb_pred
pred     = np.expm1(log_pred)
```

The pairing is deliberate rather than a default. LightGBM grows trees **leaf-wise** — always
splitting the leaf with the largest loss reduction — while XGBoost grows **level-wise**,
expanding every node at a depth before descending. Because the two build structurally different
trees, their errors fall on different rows, which is the condition under which averaging helps.
The 0.8/0.2 weighting favours the stronger model while retaining XGBoost's tighter fit
(gap 0.053) as a stabiliser. The ensemble pair gave a 0.199 RMSLE on the test dataset.

---

## Repository contents

```
notebook.ipynb    Full analysis — EDA, preprocessing, five models, ensemble
submission.csv    Final predictions
```

---

## Known limitations

Documented honestly rather than omitted.

- **CatBoost result is not a fair comparison.** Its `ColumnTransformer` omits the numeric
  block, and `ColumnTransformer` defaults to `remainder="drop"`, so all numeric features were
  discarded. Its categoricals were also pre-ordinal-encoded, bypassing CatBoost's native
  categorical handling — the mechanism that distinguishes it.
- **Predictions are not clipped to the observed price bounds** (7,500–142,000), despite the EDA
  establishing that the target is bounded. A one-line `np.clip` would guarantee no prediction
  falls outside the achievable range.
- **Mode imputation discards the missingness signal** identified in the EDA. LightGBM, XGBoost
  and HistGradientBoosting all handle `NaN` natively by learning a default split direction, so
  passing raw missing values would likely improve on imputation. The shared imputer exists
  because the decision tree baseline cannot accept `NaN`.
- **`AssetID` is left as a numeric column** and median-imputed, despite being an identifier —
  the same fix applied to `ProductConfigID` was not applied here.
- **Columns above 99.5% missing are identified in the EDA but not dropped** from the final
  feature matrix.

---

## Stack

Python · pandas · NumPy · scikit-learn · LightGBM · XGBoost · CatBoost · Matplotlib · Seaborn
