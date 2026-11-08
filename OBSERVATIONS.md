Class imbalance: click rate is 4.5% and booking rate is 2.8%, so about 97% of rows are “nothing happened”. A model that always predicts 0 would be 97% accurate and useless. That’s why you use a ranking metric (NDCG), not accuracy.

About 70% of searches end in a booking, and about 29% of rows are from random-order searches. That’s plenty for the unbiased evaluation.
Position bias is proven. In random order, the hotels at each position are random, so they’re equally good on average. Yet clicks still drop from 14% at position 1 to about 2% at position 35. That drop is purely about position.

The gap at the top shows ranking value. Expedia’s ranking gets 21.5% at position 1 vs 14.3% for random. That gap is what good ranking adds, and it’s what your model has to beat.

Dips at 5, 11, 17, 23 (every 6 positions) appear in both lines, so they’re not about hotel quality. They’re most likely a page-layout effect, such as an ad or banner slot. Good catch to mention in an interview.

Sample is representative: label distribution matches the full 9.9M rows.

Search 25 has 33 hotels. price_pct_rank climbs in steps of 1/32 = 0.03125 from the cheapest (0.0). price_vs_median of 0.51 means the cheapest hotel is about half the median price (about $53). And look at the booked hotel (relevance 5): $42, cheaper than the median (0.79), with stars above the search average (+1.09). Cheap and good is exactly the pattern the model should learn.

Location is the single strongest signal, ahead of price and reviews. Users care most about where the hotel is.
No single feature is enough, which is the case for a model that combines them.
Target: beat 0.278 clearly, and ideally get close to Expedia’s 0.378.