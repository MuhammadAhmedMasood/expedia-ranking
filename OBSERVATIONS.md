Class imbalance: click rate is 4.5% and booking rate is 2.8%, so about 97% of rows are “nothing happened”. A model that always predicts 0 would be 97% accurate and useless. That’s why you use a ranking metric (NDCG), not accuracy.

About 70% of searches end in a booking, and about 29% of rows are from random-order searches. That’s plenty for the unbiased evaluation.
Position bias is proven. In random order, the hotels at each position are random, so they’re equally good on average. Yet clicks still drop from 14% at position 1 to about 2% at position 35. That drop is purely about position.

The gap at the top shows ranking value. Expedia’s ranking gets 21.5% at position 1 vs 14.3% for random. That gap is what good ranking adds, and it’s what your model has to beat.

Dips at 5, 11, 17, 23 (every 6 positions) appear in both lines, so they’re not about hotel quality. They’re most likely a page-layout effect, such as an ad or banner slot. Good catch to mention in an interview.