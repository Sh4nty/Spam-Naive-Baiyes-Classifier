# Methodology and threat-model contract

## Research question

The project measures when a from-scratch Multinomial or Bernoulli Naive Bayes
spam filter fails under realistic distribution shift, adversarial content
modification, Unicode/HTML encoding, and training-data poisoning. It also
measures whether lightweight defences recover useful detection without
silently changing the task.

The positive class is **spam (1)** throughout. All reported precision, recall,
F1, accuracy, false-positive rate, confusion counts, attack success rates
(ASR), and confidence intervals use that convention.

## Data protocol

The victim is trained on the oldest 70% of each Enron-Spam class and tested on
the newest 30% of each class. This avoids the otherwise degenerate global
chronological split, where Enron ham ends much earlier than Enron spam.
SpamAssassin is held out as an external corpus. Five-token shingles with exact
Jaccard confirmation remove near duplicates of the victim training set from
every evaluation corpus.

No test labels are used for model fitting, feature selection, RONI acceptance,
or attack construction. The A2 attacker sees its own public ham/spam pools;
A3 sees victim weights by definition; A4 receives only the declared query
oracle.

## Models

Both classifiers use a sorted training vocabulary and add-alpha smoothing
(alpha = 1). Scores are computed in log space:

```text
S(d) = log P(spam, d) - log P(ham, d)
```

Multinomial NB counts token occurrences. Bernoulli NB counts term presence and
includes absent-term evidence. The implementation exposes the complete linear
form (`bias_`, `weights_`) so every attack and explanation can be audited
against the model's actual decision function.

## Attack and defence matrix

| ID | Capability tested | Defence/measurement |
|---|---|---|
| A1 | Oblivious random/common padding control | Clean-vs-attacked curve |
| A2 | Gray-box ham vocabulary estimate | D2 term cap, D3 clipping, D4 segments |
| A3 | White-box optimal negative-weight padding/repetition | Analytic minimum budget |
| A4 | Label-only and score-based active queries | Oracle type, per-message/total queries, ASR, recall/F1 |
| A5 | Hidden HTML text | D1 visible-text sanitisation |
| A6 | Unicode homoglyph/zero-width obfuscation | D1 Unicode and HTML normalisation |
| A7 | Defanged LLM/rule-based camouflage | Attribution of removed/added evidence |
| A8 | Ham-vocabulary spam-labelled poisoning | D5 RONI-lite and D6 retraining |
| S2 | Clipped-margin token-edit guarantee | Certified radius and certified accuracy; not semantic or arbitrary-edit robustness |

Every attack is required to preserve message count and order. A4 reports a
strict binary label oracle separately from a numeric score oracle; its query
cap is per spam message, and total queries are also recorded. D6 rebuilds the
white-box attack against the model fitted at each retraining round, then
evaluates the final model against an adaptive final-model attack.

The evaluation
harness applies attacks only to spam, leaving ham unchanged, and reports the
attackable denominator explicitly. This prevents an attack from receiving
credit for “evading” a message the clean classifier already missed.

## Statistical reporting

Point estimates are accompanied by 95% percentile bootstrap intervals (1,000
resamples, seed 42) unless a result is explicitly a fast grid sweep. Paired
before/after comparisons use exact two-sided McNemar tests where applicable.
Zero denominators are reported as zero by policy and are never emitted as
NaN. Empty evaluation sets and malformed inputs are rejected.

## Reproducibility and limitations

Run `conda activate deeplearning`, install `requirements.txt`, fetch the raw
corpora, and execute `python scripts/reproduce.py`. Raw downloaded data is
gitignored; its provenance and licensing constraints are in
[`dataset_datasheet.md`](dataset_datasheet.md). External-corpus performance is
not a claim of deployment performance. The simple tokenizer, corpus age,
language/domain assumptions, and synthetic attack budgets limit
generalisation. “Certified radius” is a guarantee for the clipped score and
token-edit model, not a guarantee against arbitrary semantic rewrites.
