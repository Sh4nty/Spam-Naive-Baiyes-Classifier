# A7 LLM camouflage: rule / rule-based-v1

Source: `data/a7/generations_rule-rule-based-v1.jsonl` (prompt v1, temperature 0.7). Test set: 4,961 Enron test ham + 300 sampled spam that both victims flag when clean (defanged, so the 'before' column already includes any effect of defanging). Refused rewrites keep the original message.

## Variant: paraphrase

- Rewrites: 300/300; refused: 0 (0.0%)
- Placeholders kept (pitch still actionable): 100.0%
- Mean normalised token edit distance: 0.06; mean length change: +5%

| Model | Precision before | after | Recall before | after | F1 before | after | ASR |
|---|---|---|---|---|---|---|---|
| multinomial | 0.615 [0.573, 0.659] | 0.615 [0.573, 0.659] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.761 [0.728, 0.795] | 0.761 [0.728, 0.795] | 0.000 |
| bernoulli | 0.809 [0.770, 0.848] | 0.803 [0.762, 0.844] | 1.000 [1.000, 1.000] | 0.963 [0.940, 0.983] | 0.894 [0.870, 0.918] | 0.876 [0.848, 0.901] | 0.037 |

**Why 11 messages evaded bernoulli NB** (summed change in log-odds contribution across evaded messages):

- Spam evidence the rewrite removed: guaranteed (2), unbelievable (2), discounts (2), limited (1), free (1)
- Ham evidence the rewrite added: chris (8), jordan (7), alex (6), thanks (5), morning (4), cheers (4), sam (3), regards (2), there (2), hi (1), good (1), could (1)

## Variant: paraphrase_pad

- Rewrites: 300/300; refused: 0 (0.0%)
- Placeholders kept (pitch still actionable): 100.0%
- Mean normalised token edit distance: 0.23; mean length change: +32%

| Model | Precision before | after | Recall before | after | F1 before | after | ASR |
|---|---|---|---|---|---|---|---|
| multinomial | 0.615 [0.573, 0.659] | 0.605 [0.562, 0.651] | 1.000 [1.000, 1.000] | 0.960 [0.937, 0.980] | 0.761 [0.728, 0.795] | 0.742 [0.708, 0.776] | 0.040 |
| bernoulli | 0.809 [0.770, 0.848] | 0.758 [0.711, 0.805] | 1.000 [1.000, 1.000] | 0.740 [0.690, 0.788] | 0.894 [0.870, 0.918] | 0.749 [0.710, 0.786] | 0.260 |

**Why 12 messages evaded multinomial NB** (summed change in log-odds contribution across evaded messages):

- Spam evidence the rewrite removed: cheap (3), limited (2), amazing (2), guaranteed (2), unbelievable (2), click (2), free (1)
- Ham evidence the rewrite added: meeting (26), schedule (26), afternoon (20), thursday (17), chris (15), week (13), team (12), updated (11), the (8), on (8), i (8), will (7)

**Why 78 messages evaded bernoulli NB** (summed change in log-odds contribution across evaded messages):

- Spam evidence the rewrite removed: guaranteed (17), discounts (8), cheap (7), free (5), limited (5), unbelievable (4), click (3), lowest (3), amazing (2), unsubscribe (2), fantastic (1), attention (1)
- Ham evidence the rewrite added: week (81), thanks (69), discuss (65), schedule (62), on (59), chris (58), meeting (57), afternoon (53), jordan (53), i (46), will (45), monday (45)
