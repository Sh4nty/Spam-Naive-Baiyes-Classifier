# Extended experiments

All results use the prepared Enron temporal-per-class split. A4, D6, S2 and the selector grid are evaluated on the untouched test set.

## A4 active black-box attack

| Model | Oracle | Total queries | Queries/message | ASR | Evaded/attackable | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| multinomial | label | 476639 | 95.8 | 0.015 | 71/4834 | 0.957 | 0.960 |
| multinomial | score | 497700 | 100.0 | 0.015 | 71/4834 | 0.957 | 0.960 |
| bernoulli | label | 410252 | 82.4 | 0.044 | 187/4277 | 0.822 | 0.895 |
| bernoulli | score | 497700 | 100.0 | 0.044 | 187/4277 | 0.822 | 0.895 |

## D6 adaptive attack-aware retraining

| Model | Rounds | Rows after | Clean F1 | Adaptive A3 F1 | Adaptive ASR | Recall |
|---|---:|---:|---:|---:|---:|---:|
| multinomial | 2 | 59657 | 0.967 | 0.734 | 0.381 | 0.603 |
| bernoulli | 2 | 59657 | 0.932 | 0.533 | 0.584 | 0.368 |

## S2 certified radius

| Model | Tau | Median radius | R@0 | R@1 | R@5 |
|---|---:|---:|---:|---:|---:|
| multinomial | 0.25 | 30.0 | 0.953 | 0.945 | 0.895 |
| multinomial | 0.50 | 28.0 | 0.956 | 0.950 | 0.905 |
| multinomial | 1.00 | 25.0 | 0.961 | 0.954 | 0.908 |
| multinomial | 2.00 | 20.0 | 0.964 | 0.954 | 0.889 |
| bernoulli | 0.25 | 46.0 | 0.570 | 0.568 | 0.557 |
| bernoulli | 0.50 | 28.0 | 0.694 | 0.686 | 0.643 |
| bernoulli | 1.00 | 19.0 | 0.826 | 0.810 | 0.745 |
| bernoulli | 2.00 | 15.0 | 0.897 | 0.878 | 0.799 |

## Feature-selection grid

| Model | Selector | k | Clean F1 | Attack F1 | ASR |
|---|---|---:|---:|---:|---:|
| multinomial | none | all | 0.967 | 0.240 | 0.854 |
| multinomial | frequency | 25 | 0.723 | 0.259 | 0.737 |
| multinomial | frequency | 100 | 0.816 | 0.059 | 0.955 |
| multinomial | frequency | 300 | 0.905 | 0.062 | 0.961 |
| multinomial | frequency | 1000 | 0.959 | 0.094 | 0.947 |
| multinomial | mi | 25 | 0.820 | 0.003 | 0.998 |
| multinomial | mi | 100 | 0.878 | 0.018 | 0.988 |
| multinomial | mi | 300 | 0.936 | 0.075 | 0.956 |
| multinomial | mi | 1000 | 0.961 | 0.168 | 0.901 |
| multinomial | chi2 | 25 | 0.822 | 0.003 | 0.998 |
| multinomial | chi2 | 100 | 0.877 | 0.018 | 0.988 |
| multinomial | chi2 | 300 | 0.938 | 0.077 | 0.955 |
| multinomial | chi2 | 1000 | 0.961 | 0.174 | 0.897 |
| bernoulli | none | all | 0.917 | 0.141 | 0.910 |
| bernoulli | frequency | 25 | 0.611 | 0.000 | 1.000 |
| bernoulli | frequency | 100 | 0.775 | 0.000 | 1.000 |
| bernoulli | frequency | 300 | 0.826 | 0.000 | 1.000 |
| bernoulli | frequency | 1000 | 0.879 | 0.022 | 0.986 |
| bernoulli | mi | 25 | 0.814 | 0.000 | 1.000 |
| bernoulli | mi | 100 | 0.846 | 0.000 | 1.000 |
| bernoulli | mi | 300 | 0.879 | 0.018 | 0.988 |
| bernoulli | mi | 1000 | 0.910 | 0.049 | 0.970 |
| bernoulli | chi2 | 25 | 0.809 | 0.000 | 1.000 |
| bernoulli | chi2 | 100 | 0.844 | 0.000 | 1.000 |
| bernoulli | chi2 | 300 | 0.877 | 0.018 | 0.989 |
| bernoulli | chi2 | 1000 | 0.909 | 0.050 | 0.969 |
