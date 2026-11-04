# M5 integrated attack and defence results

Enron temporal-per-class test split; spam is the positive class. Headline metrics include percentile bootstrap CIs; RONI uses a validation split from training and the final Enron test set only for evaluation.

## A5/A6 and D1; A2/A3 and D2-D4

| Model | Experiment | Defence | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|
| multinomial | A5 | none | 0.963 | 0.975 | 0.969 [0.966, 0.972] |
| multinomial | A5 | D1 | 0.960 | 0.972 | 0.966 [0.963, 0.969] |
| multinomial | A6 | none | 0.956 | 0.813 | 0.878 [0.871, 0.886] |
| multinomial | A6 | D1 | 0.960 | 0.972 | 0.966 [0.963, 0.969] |
| multinomial | A2 | D2 | 0.979 | 0.973 | 0.976 [0.973, 0.979] |
| multinomial | A2 | D3 | 0.960 | 0.980 | 0.970 [0.967, 0.973] |
| multinomial | A2 | D4 | 0.848 | 0.984 | 0.911 [0.905, 0.916] |
| multinomial | A3 | D2 | 0.794 | 0.081 | 0.147 [0.135, 0.159] |
| multinomial | A3 | D3 | 0.931 | 0.555 | 0.695 [0.683, 0.707] |
| multinomial | A3 | D4 | 0.821 | 0.810 | 0.815 [0.807, 0.823] |
| bernoulli | A5 | none | 0.983 | 0.845 | 0.909 [0.903, 0.915] |
| bernoulli | A5 | D1 | 0.984 | 0.838 | 0.905 [0.899, 0.911] |
| bernoulli | A6 | none | 0.918 | 0.159 | 0.271 [0.255, 0.285] |
| bernoulli | A6 | D1 | 0.984 | 0.838 | 0.905 [0.899, 0.911] |
| bernoulli | A2 | D2 | 0.984 | 0.868 | 0.922 [0.917, 0.928] |
| bernoulli | A2 | D3 | 0.984 | 0.738 | 0.844 [0.836, 0.852] |
| bernoulli | A2 | D4 | 0.961 | 0.856 | 0.905 [0.899, 0.911] |
| bernoulli | A3 | D2 | 0.844 | 0.077 | 0.141 [0.129, 0.153] |
| bernoulli | A3 | D3 | 0.941 | 0.187 | 0.312 [0.297, 0.327] |
| bernoulli | A3 | D4 | 0.953 | 0.699 | 0.807 [0.798, 0.816] |

## A8 poisoning

| Model | Rate | Count | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| multinomial | 0.5% | 118 | 0.963 | 0.971 | 0.967 |
| multinomial | 1.0% | 236 | 0.962 | 0.971 | 0.967 |
| multinomial | 2.0% | 472 | 0.962 | 0.972 | 0.967 |
| multinomial | 5.0% | 1180 | 0.962 | 0.971 | 0.967 |
| bernoulli | 0.5% | 118 | 0.984 | 0.866 | 0.921 |
| bernoulli | 1.0% | 236 | 0.985 | 0.875 | 0.926 |
| bernoulli | 2.0% | 472 | 0.985 | 0.898 | 0.939 |
| bernoulli | 5.0% | 1180 | 0.982 | 0.941 | 0.961 |

## D5 RONI-lite

RONI decisions use a validation split from the training set; the final test set is not used for acceptance decisions.

| Model | Candidates | Accepted | Rejected | Threshold | Validation F1 before | Validation F1 after | Test F1 after |
|---|---:|---:|---:|---:|---:|---:|---:|
| multinomial | 20 | 20 | 0 | 0.010 | 0.985 | 0.985 | 0.967 |
| bernoulli | 20 | 20 | 0 | 0.010 | 0.969 | 0.969 | 0.938 |
