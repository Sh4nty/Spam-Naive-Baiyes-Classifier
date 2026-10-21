# Good-word attacks A1-A3 (Enron test set)

Victims trained on the Enron training split, full vocabulary, alpha = 1. Attacks append words to every spam message in the Enron test set; ham is untouched. Budget = tokens appended per message. ASR is measured on the spam each model flags when clean. A1-random is averaged over 5 seeds; the other attacks are deterministic. A1 and A2 draw only on the attacker's own corpora (SpamAssassin ham and spam, disjoint from the victim's training data); A3 reads the victim's weights.

## F1 by budget

### bernoulli (clean F1 = 0.917)

| Attack | b=0 | b=5 | b=10 | b=20 | b=50 | b=100 | b=200 | b=500 | Breaking point | Robustness AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 random padding (T1, control) | 0.917 | 0.928 | 0.939 | 0.956 | 0.982 | 0.991 | 0.993 | 0.993 | > 500 | 1.078 |
| A1 common-word padding (T1, control) | 0.917 | 0.905 | 0.887 | 0.893 | 0.896 | 0.992 | 0.993 | 0.993 | > 500 | 1.066 |
| A2 passive good-word (T2) | 0.917 | 0.935 | 0.926 | 0.922 | 0.850 | 0.768 | 0.791 | 0.993 | > 500 | 0.940 |
| A3 white-box, distinct words (T4) | 0.917 | 0.468 | 0.280 | 0.141 | 0.042 | 0.005 | 0.000 | 0.000 | 10 | 0.025 |

### multinomial (clean F1 = 0.967)

| Attack | b=0 | b=5 | b=10 | b=20 | b=50 | b=100 | b=200 | b=500 | Breaking point | Robustness AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 random padding (T1, control) | 0.967 | 0.970 | 0.972 | 0.975 | 0.979 | 0.981 | 0.981 | 0.981 | > 500 | 1.014 |
| A1 common-word padding (T1, control) | 0.967 | 0.967 | 0.965 | 0.968 | 0.972 | 0.981 | 0.981 | 0.981 | > 500 | 1.013 |
| A2 passive good-word (T2) | 0.967 | 0.972 | 0.968 | 0.967 | 0.937 | 0.851 | 0.794 | 0.980 | > 500 | 0.912 |
| A3 white-box, distinct words (T4) | 0.967 | 0.688 | 0.387 | 0.240 | 0.101 | 0.042 | 0.008 | 0.000 | 10 | 0.046 |
| A3 white-box, one word repeated (T4) | 0.967 | 0.564 | 0.310 | 0.193 | 0.053 | 0.008 | 0.000 | 0.000 | 10 | 0.029 |

## Headline: budget 50 with 95% bootstrap CIs

| Model | Attack | Recall | F1 | ASR |
|---|---|---|---|---|
| multinomial | A1 random padding (T1, control) | 0.996 [0.994, 0.997] | 0.979 [0.977, 0.982] | 0.001 |
| multinomial | A1 common-word padding (T1, control) | 0.981 [0.977, 0.985] | 0.972 [0.969, 0.975] | 0.000 |
| multinomial | A2 passive good-word (T2) | 0.915 [0.907, 0.923] | 0.937 [0.932, 0.942] | 0.058 |
| multinomial | A3 white-box, distinct words (T4) | 0.055 [0.049, 0.061] | 0.101 [0.090, 0.112] | 0.943 |
| multinomial | A3 white-box, one word repeated (T4) | 0.028 [0.023, 0.033] | 0.053 [0.044, 0.061] | 0.971 |
| bernoulli | A1 random padding (T1, control) | 0.978 [0.974, 0.982] | 0.982 [0.979, 0.984] | 0.000 |
| bernoulli | A1 common-word padding (T1, control) | 0.822 [0.812, 0.832] | 0.896 [0.889, 0.902] | 0.049 |
| bernoulli | A2 passive good-word (T2) | 0.750 [0.738, 0.762] | 0.850 [0.842, 0.858] | 0.128 |
| bernoulli | A3 white-box, distinct words (T4) | 0.021 [0.018, 0.026] | 0.042 [0.034, 0.050] | 0.975 |

## Analytic minimum effort b*(d) on the attackable spam

b*(d) = fewest appended good words that flip message d, computed from the true weights (distinct: greedy over the most negative w_t; repeat: ceil(S(d)/|w_min|)). The tests check it is exact.

| Model | Mode | Median | 10th pct | 90th pct | Flippable with <= 50 | <= 500 |
|---|---|---|---|---|---|---|
| multinomial | distinct | 6 | 2 | 34 | 94.3% | 100.0% |
| multinomial | repeat | 5 | 2 | 22 | 97.1% | 100.0% |
| bernoulli | distinct | 4 | 1 | 19 | 97.5% | 100.0% |

## What each attacker picks as 'good words'

- **multinomial**, A3 white-box top 10: enron, kaminski, ect, ees, mmbtu, ena, enronxgate, stinson, cera, sitara
- **multinomial**, A2 passive top 10 with the victim's true weight (negative = hammy): freshrpms (not in vocab), redhat (+2.73), newsisfree (not in vocab), razor (+2.03), wrote (-2.07), exmh (+0.65), rpm (+1.42), matthias (-1.15), satalk (+0.65), docs (-3.54)
- **bernoulli**, A3 white-box top 10: enron, kaminski, mmbtu, ena, stinson, gibner, sitara, enronxgate, ect, ees
- **bernoulli**, A2 passive top 10 with the victim's true weight (negative = hammy): freshrpms (not in vocab), redhat (+2.74), newsisfree (not in vocab), razor (+2.04), wrote (-1.95), exmh (+0.66), rpm (+1.43), matthias (-1.14), satalk (+0.66), docs (-3.13)
