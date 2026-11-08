# Hotel Search Ranking: Learning-to-Rank on Expedia Search Data

Ranking hotels within a search so the ones a user is most likely to book appear at the top. Built on ~10M real Expedia search impressions, with search-relative feature engineering in SQL, LightGBM LambdaRank, and position-bias-aware evaluation on randomised traffic.

**Headline result:** LightGBM LambdaRank reaches **NDCG@5 = 0.367** on a held-out future period, **+32% over the best non-ML rule** (rank by location score, 0.278). On randomised-order traffic, where position bias cannot help, it is still **+23% over that rule** and beats the displayed order.

---

## 1. Business problem

When a user searches for "Paris, 3 nights, 2 adults", the site has ~25–40 matching hotels and must decide the order to show them. Users mostly look at the first few results, so the order largely decides whether a booking happens. Better ranking means more bookings per search and more revenue.

This mirrors the core problem for any online travel agent: ranking packages/hotels on a search results page.

## 2. Data

- Source: Kaggle *Personalize Expedia Hotel Searches – ICDM 2013* (`train.csv`)
- 9,917,530 rows (one row = one hotel shown in one search), 399,344 searches, Nov 2012 – Jun 2013
- Per row: hotel attributes (stars, review score, location scores, price, promotion), search context (length of stay, booking window, party size), visitor history, competitor price comparisons, display position, and outcomes (click, booking)
- About 29% of searches were shown in **random order** by Expedia (`random_bool = 1`), which enables unbiased evaluation

**Working sample:** 50,000 searches (1,239,782 rows), sampled by `srch_id` so every result list stays intact. The label distribution matches the full data.

**Key EDA findings**
- Strong class imbalance: 4.5% of rows clicked, 2.8% booked; 69.5% of searches end in a booking (about 1 booking among ~25 hotels)
- **Position bias:** in random-order searches, click rate still falls from 14% at position 1 to ~2% at position 35, purely because of position
- Regular dips at positions 5, 11, 17, 23 in both ranked and random traffic, likely a page-layout effect
- Data quality: missing values were encoded as the text `"NULL"`; fixed at load time (`nullstr='NULL'`). Visitor history is ~95% missing, competitor columns 52–98% missing, and `price_usd` has extreme outliers (max ≈ $19.7M)

## 3. Approach

### Label
Graded relevance: **5 = booked, 1 = clicked only, 0 = neither** (the competition's official scheme). LambdaRank uses gain = 2^label − 1, so a booking is worth 31× a click, reflecting that bookings drive revenue.

### Leakage control
Rule: *only use information available at the moment of ranking.*
- Excluded: `click_bool`, `booking_bool`, `gross_bookings_usd` (outcomes), `position` (output of Expedia's own ranker, unknown before ranking), `random_bool` (display mode, used only for evaluation), raw IDs (`srch_id`, `prop_id`, `srch_destination_id`)
- Deliberately **not** built: hotel historical click/booking rates (target encoding). Highly predictive, but computed naively they leak the target; doing it properly requires a strictly time-based calculation (listed as a next step)

### Feature engineering (SQL, DuckDB)
Users compare hotels **within one result list**, so absolute values mislead across searches ($150 is cheap in London, expensive elsewhere). The main features are therefore **relative to the search**, built with window functions partitioned by `srch_id`:

| Group | Features |
|---|---|
| Price cleaning | `price_per_night` (price conventions differ by site), `log_price` (outliers) |
| Hotel vs its own search | `price_pct_rank`, `price_vs_median` (median is robust to outliers), `star_vs_mean`, `review_vs_mean`, `loc2_vs_mean`, `n_props_in_search` |
| Deal & personalisation | `hist_price_diff` (usual price − today's price; NULL when no history), `visitor_star_diff`, `visitor_price_diff` |
| Competitors | 24 sparse columns summarised into `comp_cheaper_cnt`, `comp_dearer_cnt`, `comp_known_cnt` |

35 features in total (engineered + original hotel/search attributes; site and country IDs treated as categorical).

### Split
**Time-based**, by search date: train 70% (Nov 1 – Apr 29), validation 10% (Apr 29 – May 21), test 20% (May 21 – Jun 30). A random split would leak future patterns (seasonality, price trends). Validation is used only for early stopping; the test set was never used for any decision.

### Metric
**NDCG@5** per search, averaged over test searches (searches with no click excluded). Accuracy is meaningless at 97% negatives; precision/recall/F1 need an arbitrary threshold we never use; AUC compares hotels across different searches and weights all positions equally. NDCG@5 scores the order *within* each search and focuses on what users actually see.

### Models
| Model | Type | Why included |
|---|---|---|
| Logistic regression | Pointwise, linear | Linear baseline (with median imputation + scaling in a pipeline) |
| LightGBM classifier | Pointwise, trees | Predicts P(booking); captures nonlinearity and interactions, handles NULLs natively |
| **LightGBM LambdaRank** | Listwise, trees | Optimises NDCG directly by comparing hotels within each search |

Hyperparameters: standard values with mild regularisation for rare positives (`num_leaves=63`, `min_child_samples=100`, `colsample_bytree=0.8`, `learning_rate=0.05`); number of trees chosen by early stopping on validation (classifier: 282, best validation AUC 0.781; ranker: 185).

## 4. Results

### Test set (10,000 searches, future period)

| Ranking | NDCG@5 |
|---|---|
| Random | 0.162 |
| Best reviewed first | 0.170 |
| Cheapest first | 0.209 |
| Best location first (best rule) | 0.278 |
| Logistic regression | 0.321 |
| LambdaRank, raw features only (ablation) | 0.350 |
| LightGBM classifier | 0.365 |
| **LightGBM LambdaRank** | **0.367** |
| Expedia's displayed order (reference only) | 0.378 |

### Normal vs randomised traffic

| Ranking | Normal traffic | Random-order traffic |
|---|---|---|
| Random | 0.158 | 0.171 |
| Best location first | 0.285 | 0.262 |
| Logistic regression | 0.336 | 0.288 |
| LightGBM classifier | 0.385 | 0.321 |
| **LightGBM LambdaRank** | **0.387** | **0.323** |
| Expedia's displayed order | 0.409 | 0.306 |

### Key insights
1. **+32% over the best business rule** (0.278 → 0.367) on a future time period.
2. **Feature engineering mattered more than algorithm choice.** Search-relative features added +0.018 NDCG (+5%); LambdaRank on raw features scored below the classifier with engineered features. The engineered features carry ~29% of total gain importance.
3. **Trees beat linear** (+0.046), showing that interactions matter (e.g. price sensitivity varies with context).
4. **LambdaRank only marginally beats the classifier** (+0.002). Because a booking's gain dominates NDCG, predicting P(booking) per hotel is already close to what NDCG rewards.
5. **Position bias is large and measurable.** On random-order traffic, ranking by displayed position still scores 0.306 vs 0.171 for random, because users click what is at the top. That 0.135 gap is position bias itself. Expedia's 0.378 on the full test set is inflated by the same effect.
6. **The model's skill is not borrowed from position.** It never sees position, yet on randomised traffic it beats both the displayed order (0.323 vs 0.306) and the best rule by 23%.
7. **Location is the strongest single signal** (`prop_location_score2`: 17% of gain), consistent with the location baseline being the best rule.
8. **Price:** on random-order traffic, the cheapest 10% of hotels within a search book at 0.71% vs 0.28% for the most expensive 10% (~2.5×). The model reproduces this shape but over-predicts the level (~2×), because it was trained mostly on normal traffic where booking rates are higher. That doesn't matter for ranking (only order counts), but probabilities would need recalibration before use in pricing or revenue forecasts.

## 5. Proposed A/B test

Offline NDCG is a proxy. The real test is a controlled online experiment.

- **Hypothesis:** the LambdaRank ranking increases booking conversion versus the current ranking.
- **Design:** randomise at **user** level (not search level), so a user sees a consistent ranking; 50/50 split between control (current ranking) and treatment (model).
- **Primary metric:** booking conversion per search.
- **Secondary metrics:** revenue/margin per search, click-through rate, rank of the booked item.
- **Guardrails:** page latency, cancellation rate, average booked price (to catch the model simply pushing cheap low-margin options), complaint/refund rate.
- **Sample size** (two-proportion test, α = 0.05 two-sided, 80% power), using a baseline conversion of 69.5% of searches (this dataset):
  - to detect a **1% relative lift** (69.5% → 70.2%): ~**68,400 searches per arm**
  - to detect a **2% relative lift**: ~**17,000 searches per arm**
  - Note that a real site-wide conversion rate per search is likely far lower than this dataset's 69.5%. At 3% conversion, detecting a 2% relative lift needs ~1.28M searches per arm, which is why the primary metric and MDE must be agreed with the business before launch.
- **Duration:** run for full weekly cycles (at least 2 weeks) to cover weekday/weekend behaviour, even if the sample size is reached sooner; no peeking at results before the planned end (or use a sequential testing method).
- **Analysis:** because randomisation is by user but the metric is per search, use cluster-robust standard errors (or the delta method); consider CUPED with pre-period behaviour to reduce variance.

```python
from math import ceil
z = 1.959964 + 0.841621          # alpha 0.05 two-sided, power 0.80
p1 = 0.695; p2 = p1 * 1.01       # baseline, 1% relative lift
n = z**2 * (p1*(1-p1) + p2*(1-p2)) / (p2 - p1)**2
print(ceil(n))                   # ~68,400 per arm
```

## 6. Limitations

- **Offline evaluation only.** Logged data reflects the old ranking (feedback loop); only an A/B test proves business impact.
- **Biased labels on normal traffic.** Clicks happen where items are shown. Random traffic mitigates this but is noisier (fewer bookings).
- **Price is confounded with quality.** The price curve is not a true price elasticity; that requires randomised prices.
- **Possible over-reliance on country IDs.** `prop_country_id` and `visitor_location_country_id` rank 2nd and 3rd by gain. High-cardinality categoricals inflate gain importance and may partly memorise; needs an ablation or stronger categorical regularisation.
- **Sample and age.** Trained on a 50k-search sample of 2013 data; behaviour and the market have changed.
- **Limited tuning.** Hyperparameters were not systematically searched.
- **`price_usd` conventions** differ by site (per night vs per stay); `price_per_night` only partially corrects this.

## 7. Next steps

1. **Time-aware target encoding:** hotel historical click/booking rates computed only from data before each search
2. **Position-bias correction in training:** inverse propensity weighting, or LightGBM's unbiased LambdaRank (`lambdarank_unbiased`)
3. **Hyperparameter tuning** with Optuna against validation NDCG@5; compare with XGBoost `rank:ndcg`
4. **Ablation of categorical IDs** and regularisation (`min_data_per_group`, `cat_smooth`)
5. **Train on the full 400k searches**
6. **Serving:** FastAPI scoring endpoint; monitoring of feature drift (PSI) and prediction drift; scheduled retraining
7. **Pricing extension:** recalibrate probabilities (isotonic regression) and estimate expected margin per hotel to combine relevance with commercial value

## 8. Repository structure

```
expedia-ranking/
├── 01_eda.ipynb          # load raw CSV, sample 50k searches, write Parquet
├── 02_eda.ipynb          # EDA (rates, nulls, position bias), label, SQL feature engineering
├── 03_model.ipynb        # time split, baselines, models, evaluation, feature importance
├── DECISIONS.md          # design decisions and alternatives considered
├── OBSERVATIONS.md       # findings from each step
└── data/                 # not committed (see below)
```

## 9. How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install duckdb pandas lightgbm scikit-learn matplotlib jupyter ipykernel
```
1. Download `train.csv` from the Kaggle competition *Personalize Expedia Hotel Searches – ICDM 2013* and place it in `data/` (on macOS, unzip with the `unzip` command; Archive Utility can't open the Deflate64 archive).
2. Run the notebooks in order: `01_eda` → `02_eda` → `03_model`.

**Tech:** Python, DuckDB (SQL), pandas, NumPy, LightGBM, scikit-learn, matplotlib.