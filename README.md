# Where Naive Bayes Breaks (Group 2)

From-scratch Multinomial and Bernoulli Naive Bayes (Manning et al., Ch. 13)
in NumPy/Pandas, stress-tested with good-word, encoding, LLM-camouflage and
poisoning attacks. scikit-learn is never imported.

## Setup

```
pip install -r requirements.txt
python -m pytest -q              # all tests should pass
python scripts/fetch_data.py     # Enron-Spam + SpamAssassin into data/raw/ (gitignored)
```

## Reproduce

| Command | Output |
|---|---|
| `python scripts/run_baseline.py` | `results/baseline.md`: clean P/R/F1/FPR with 95% CIs |
| `python scripts/run_attacks.py` | `results/attacks_a1_a3.md`, `.csv`, `attacks_f1_vs_budget.png` (~7 min) |
| `python scripts/run_m5.py` | `results/m5_integrated.md` and `.json`: A5/A6/D1, A8, and D2-D5 outputs |
| `python scripts/prepare_a7.py` | `data/a7/a7_sample.jsonl`: 300 defanged spam for A7 (already committed) |
| `python scripts/generate_a7.py --backend ...` | LLM rewrites; see the script's docstring for Ollama / Groq / Gemini / rule-based |
| `python scripts/evaluate_a7.py` | `results/a7_<model>.md`: before/after P/R/F1, ASR, token attribution |

The first script run builds `data/processed/` (cached, about 30 s).

## Code map (`src/spam_nb/`)

| Module | Contents |
|---|---|
| `tokenizer.py` | Lowercase regex tokenizer |
| `naive_bayes.py` | `MultinomialNB`, `BernoulliNB`; `weights_`, `bias_`, `margin()`, `token_contributions()`; optional `selector`/`k` |
| `features.py` | Frequency, mutual information and chi-square selection (IIR section 13.5) |
| `data.py` | Enron and SpamAssassin loaders, temporal split, near-duplicate removal |
| `metrics.py` | P/R/F1/FPR with spam positive, bootstrap CIs, `evaluate(model, docs, labels, attack)` |
| `attacks.py` | A1 padding, A2 passive good-word, A3 white-box, analytic bound `min_good_words()` |
| `llm_camouflage.py` | A7 defanging, prompts, rule-based fallback, evasion attribution |
| `encoding.py` | A5 hidden HTML, A6 Unicode obfuscation, D1 input sanitisation |
| `poisoning.py` | A8 ham-vocabulary poison generation and training-data injection |
| `defences.py` | D2 term-frequency caps, D3 weight clipping, D4 segment scoring, D5 RONI-lite |
| `experiment.py` | Shared config, prepared data and victim fitting for the scripts |

## Writing an attack

An attack is any function from a list of spam messages to the modified list
(same length and order). Pass it to `evaluate()`:

```python
from spam_nb.experiment import load_config, prepare_data, fit_victim
from spam_nb.metrics import evaluate

cfg = load_config(); data = prepare_data(cfg)
model = fit_victim("multinomial", data, cfg)
test = data["enron_test"]
res = evaluate(model, test.text, test.label, attack=lambda msgs: [m + " enron" for m in msgs])
res["attacked"]["metrics"], res["asr"]
```

## Protocol notes

- **Split.** Enron-Spam ham ends in 2002 while its spam runs to 2005, so a
  single "oldest 70% / newest 30%" cut gives a test set that is 100% spam.
  Each class is split at its own date instead (`split_strategy:
  temporal_per_class` in `config/seed.json`).
- **De-duplication.** Near-duplicates (5-word shingles, Jaccard >= 0.8) of
  Enron training messages are removed from every test set. About 43% of
  SpamAssassin spam duplicates Enron training spam, because Enron-Spam's
  spam was partly drawn from the SpamAssassin corpus.
- **Domain shift.** Models trained on Enron score poorly on SpamAssassin ham
  (mailing-list vocabulary appears only in Enron spam). Attack results are
  therefore reported on the Enron test set.
- **Ethics.** Attacks run offline against our own models only. URLs, e-mail
  addresses and phone numbers in A7 material are replaced by placeholders.
  Never feed corpus messages into a live mail system.
