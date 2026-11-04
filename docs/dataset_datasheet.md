# Dataset datasheet

## Enron-Spam

- **Role:** Victim training and temporally separated in-domain test set.
- **Source:** The Enron-Spam release downloaded by `scripts/fetch_data.py`.
- **Labels:** `0 = ham`, `1 = spam`.
- **Split:** The oldest 70% of each class is training data; the newest 30% is
  test data. Splitting per class avoids an all-spam test set caused by the
  different time ranges of ham and spam.
- **Deduplication:** Five-word shingle similarity with Jaccard threshold 0.8
  removes test messages near-duplicated in training data.
- **Preprocessing:** Subject and body are combined; RFC822 parts and HTML are
  parsed; tokenization is lowercase regex tokenization.

## SpamAssassin

- **Role:** External test set and attacker-side public ham/spam pools.
- **Source:** Public SpamAssassin corpus downloaded by `scripts/fetch_data.py`.
- **Privacy and safety:** Messages are used offline only. Generated outputs
  defang URLs, e-mail addresses and phone numbers.
- **Leakage control:** Messages near-duplicated in Enron training data are
  removed before external evaluation or attacker-pool construction.

## Fresh-100 and Reuters-21578

These proposal datasets are **not included in this repository snapshot**.
The project does not fabricate labels or results for them. To add either
corpus, provide a local, licensed file with the columns required by
`spam_nb.data.standardize` (`text`, `label`, `timestamp`) and record source,
licence, collection date, hashes, annotation procedure and any adjudication
statistics before evaluating it. The local reproduction command reports all
datasets that are actually present.

## Limitations

The Enron and SpamAssassin corpora are historical and domain-specific.
Results should not be interpreted as performance on current mail or as a
claim about any live filtering system.
