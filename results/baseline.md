# Clean baselines (no attack)

Spam is the positive class. Values are point estimates with 95% percentile bootstrap CIs (1000 resamples, seed 42). Smoothing alpha = 1.0, full vocabulary.

## Data

- Enron train: 23,600 messages, 50.9% spam, 1999-12-10 to 2005-06-08
- Enron test: 9,938 messages, 50.1% spam, 2001-07-09 to 2005-09-06
- SpamAssassin (external): 5,226 messages, 20.6% spam, 2001-06-25 to 2002-12-04

Split: Enron-Spam ham ends in 2002 while spam runs to 2005, so one global cut leaves a 100%-spam test set. Each class is split at its own date: oldest 70% train, newest 30% test.

Near-duplicates removed (5-word shingles, Jaccard >= 0.8): 178 Enron test messages that duplicate Enron training messages, and 820 SpamAssassin messages (820 of them spam) that duplicate Enron training messages. Enron-Spam's spam was partly drawn from the SpamAssassin corpus, so without this step SpamAssassin would not be an external test.

## Results

| Model | Test set | Precision | Recall | F1 | Accuracy | FPR | TP/FP/TN/FN |
|---|---|---|---|---|---|---|---|
| multinomial | Enron test | 0.963 [0.958, 0.968] | 0.971 [0.967, 0.976] | 0.967 [0.964, 0.970] | 0.967 [0.963, 0.970] | 0.038 [0.033, 0.043] | 4834/188/4773/143 |
| multinomial | SpamAssassin (external) | 0.235 [0.223, 0.246] | 0.981 [0.974, 0.989] | 0.379 [0.363, 0.394] | 0.339 [0.327, 0.352] | 0.827 [0.815, 0.839] | 1056/3434/716/20 |
| bernoulli | Enron test | 0.984 [0.980, 0.987] | 0.859 [0.850, 0.869] | 0.917 [0.912, 0.923] | 0.922 [0.917, 0.927] | 0.014 [0.011, 0.018] | 4277/71/4890/700 |
| bernoulli | SpamAssassin (external) | 0.288 [0.273, 0.303] | 0.880 [0.859, 0.900] | 0.434 [0.415, 0.451] | 0.527 [0.514, 0.541] | 0.564 [0.548, 0.579] | 947/2341/1809/129 |
