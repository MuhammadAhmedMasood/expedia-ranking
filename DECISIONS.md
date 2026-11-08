## Sampling: 
sampled 50k of 399k searches by srch_id (reservoir, seed 42), keeping whole searches intact because ranking needs complete result lists. Saved to Parquet for speed.

## SUMMARIZE tells:
1. Visitor history (visitor_hist_starrating, visitor_hist_adr_usd): about 95% missing, since most users are new. They’re still useful when present; LightGBM handles NULLs natively.
2. srch_query_affinity_score: 94% missing. orig_destination_distance: 32% missing. prop_location_score2: 22% missing.
3. Competitor columns: 52–98% missing. Plan to summarise them into a couple of features rather than using all 24.
price_usd max = 19,726,328: an obvious outlier or data error. We’ll handle it by using price relative to the search median, plus a log transform.
4. gross_bookings_usd is 97.21% null, which matches the 2.8% booking rate exactly. That confirms it only exists when a booking happens, so it’s leakage and gets excluded.
5. prop_starrating min = 0: 0 probably means “unknown”, not zero stars.



## Label: 5 = booked, 1 = clicked only, 0 = neither

Alternatives: booking only (the true business goal, but very sparse, only 2.8% positive), or click only (more signal, but clicks aren’t revenue).
Choice: a graded label uses both. Clicks add signal, and bookings count much more. LightGBM’s LambdaRank turns the label into a gain of 2^label − 1, so a booking is worth 31 and a click is worth 1. Ranking the booked hotel high dominates the score.

## Excluded columns:

click_bool, booking_bool, gross_bookings_usd: these are the outcome. gross_bookings_usd only exists when a booking happens (97.21% null = 2.8% booked), so including it would hand the model the answer.
position: it’s Expedia’s own ranker’s output, and it only exists after ranking. In production, your model runs before positions exist. Including it would mean learning to copy Expedia’s ranking, plus position bias.
random_bool: describes how the page was displayed, not anything about the user or hotel. It’s used only to build the unbiased test slice.
IDs (srch_id, prop_id, srch_destination_id): arbitrary numbers. Feeding them in as numbers makes the model treat “hotel 140,000” as bigger than “hotel 5”, which is meaningless, and it tends to memorise individual hotels instead of learning patterns. srch_id is still used to group rows into searches, and date_time is used for the time split.

## Feature Engineering:
Features: relative to the search. Users compare hotels within one result list, so I added within-search features using window functions: price percentile, price vs search median, and star/review/location vs search mean. Alternative: raw features only. I’ll compare both, to measure the gain.

Competitors: 24 sparse columns (52–98% NULL) summarised into 3 counts (cheaper / dearer / known), which cuts noise.

Price: log transform for outliers; per-night price because price_usd conventions differ by site; median rather than mean because it’s robust to outliers.

Not done (deliberately): hotel historical click/booking rates (“target encoding”). They’re very predictive, but if you compute them on the same data you evaluate on, they leak the answer. Doing it properly needs a time-based calculation using only past data. Listed as a next step.

## Splitting data into training, validation and testing

Split: by time (70/10/20 of searches by date), not random. In production we train on the past and predict the future, and a random split would leak future patterns (seasonality, price trends) into training. The validation set is for early stopping in Step 6. The test set is untouched until final evaluation.

Metric: NDCG@5 with gain 2^rel − 1 (matches LambdaRank), with searches without any click excluded. Top-5 because users rarely scroll, and accuracy is useless at 97% zeros.

Baselines: random, cheapest-first, best-reviewed, best-location. Expedia’s own ranking is shown as a reference only, since it’s inflated by position bias on non-random traffic.

