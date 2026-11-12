# Experimental report

This is the post-run interpretation of the generated files in `results/`.
The command used was:

```text
conda activate deeplearning
python scripts/reproduce.py
```

Configuration: seed 42, temporal-per-class split, 70/30 train/test, alpha 1,
5-token-shingle near-duplicate threshold 0.8, and 1,000 bootstrap resamples
for headline estimates.

## Executive findings

1. **In-domain performance is strong but not sufficient.** Multinomial NB
   reaches F1 0.967 on Enron test; Bernoulli reaches 0.917.
2. **External domain shift dominates.** On SpamAssassin, F1 falls to 0.380
   (Multinomial) and 0.434 (Bernoulli), with false-positive rates of 0.827 and
   0.564. A model that looks excellent in-domain is not deployment-ready.
3. **White-box padding is a decisive failure mode.** At budget 50, A3 reduces
   F1 to 0.101 (Multinomial) and 0.042 (Bernoulli), with ASR 0.943 and 0.975.
   Random padding is a matched control and is substantially less damaging;
   this supports, but does not prove, the role of feature direction.
4. **Input canonicalisation is highly effective for encoding attacks.** D1
   restores most A6 performance: Multinomial F1 rises from 0.878 to 0.966;
   Bernoulli rises from 0.271 to 0.905.
5. **Simple defences have trade-offs.** D2/D3/D4 improve selected attacks but
   do not eliminate a white-box oracle. Defence selection must therefore be
   evaluated against an explicit threat model.
6. **Poisoning is not uniformly damaging at tested rates.** Ham-vocabulary
   poisoning up to 5% leaves Multinomial F1 essentially unchanged; Bernoulli
   improves on this synthetic protocol. This is evidence about this protocol,
   not proof that poisoning is safe.
7. **A4 and D6 are protocol-explicit.** A4 separates binary label-only and
   numeric score oracles and reports per-message and total query cost. D6
   rebuilds the attacker after each retraining round and evaluates the final
   model against an adaptive white-box attack. D6 retains clean F1 of 0.967
   (Multinomial) and 0.932 (Bernoulli), but adaptive A3 still reduces F1 to
   0.734 and 0.533 respectively; it is not a complete defence.

## Results by evidence file

- [`baseline.md`](../results/baseline.md): clean in-domain and external
  baselines, confusion counts, and bootstrap intervals.
- [`attacks_a1_a3.md`](../results/attacks_a1_a3.md): threat-model-separated
  padding curves, breaking points, and analytic effort bounds.
- [`m5_integrated.md`](../results/m5_integrated.md): A5/A6, D1-D5, and A8.
- [`extended_experiments.md`](../results/extended_experiments.md): A4, D6,
  certified-radius S2, and feature-selection sensitivity.
- [`a7_rule-rule-based-v1.md`](../results/a7_rule-rule-based-v1.md): the
  reproducible rule-based camouflage control and attribution.

## What a research-grade conclusion can and cannot claim

The evidence supports the claim that a transparent Naive Bayes baseline can be
excellent on a temporally held-out sample while remaining vulnerable to
domain shift, feature-direction attacks, and representation attacks. It also
supports input canonicalisation and explicit robustness reporting as
engineering requirements.

It does **not** support claims of universal adversarial robustness, production
readiness, language independence, superiority over modern neural filters, or
causal identification of why matched padding controls differ. The certified
radius applies only to the clipped score and token insertion/deletion threat
model; it does not cover paraphrase, HTML, Unicode, or semantic edits.

The defensible central claim is: **on the historical Enron-Spam corpus, a
from-scratch Naive Bayes filter achieves strong temporal holdout performance
but remains sensitive to corpus shift and white-box token-padding attacks,
while input normalization substantially mitigates the tested encoding
attacks.** These findings are protocol-specific and do not establish
robustness against adaptive semantic, contemporary, or production-scale
attacks.
The next scientifically meaningful extensions are fresh contemporary corpora,
multiple independent random seeds, calibrated probability evaluation, and
semantic-preserving human or model-reviewed attacks.

## Reproduction checklist

1. Activate `deeplearning` and verify the interpreter.
2. Run `python -m pytest -q` and record the test count.
3. Fetch or verify the raw corpora and inspect `docs/dataset_datasheet.md`.
4. Run `python scripts/reproduce.py`.
5. Compare Markdown and JSON outputs under `results/`.
6. Report exact configuration, corpus counts, duplicate-removal counts,
   confidence-interval method, and any omitted optional corpus.
