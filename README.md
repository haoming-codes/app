# ASR Post-Correction

This project provides a lightweight toolkit for correcting Chinese ASR output, with a focus on named-entity errors. The approach mimics a retrieval-augmented generation (RAG) workflow: a user-supplied lexicon of frequently mis-recognised entities is converted into IPA-like phonetic features, and the ASR transcript is searched for substrings with similar pronunciations. Candidates that fall below a configurable distance threshold are proposed as corrections.

## Features

- IPA-oriented conversion for Mandarin syllables using `pypinyin` with an explicit mapping to IPA segments.
- Phonetic similarity scoring using `panphon`'s weighted feature edit distance.
- Greedy conflict resolution to apply the best-matching, non-overlapping corrections.
- Simple API for integrating into post-processing pipelines.

## Installation

Install the package in editable mode:

```bash
pip install -e .
```

The core dependencies are [`pypinyin`](https://github.com/mozillazg/python-pinyin) and [`panphon`](https://github.com/dmort27/panphon).

## Quick start

```python
from asr_postcorrection import Lexicon, PhoneticMatcher

# Build a lexicon of problematic named entities.
lexicon = Lexicon.from_strings(["张三", "李四", "王小明"])
matcher = PhoneticMatcher(lexicon)

asr_output = "欢迎张山来到我们的节目。"
corrected, corrections = matcher.apply_best_corrections(asr_output, threshold=1.2)

print(corrected)
for candidate in corrections:
    print(candidate)
```

## Workflow overview

1. **Lexicon preparation:** curate a list of named entities, ideally the phrases that are frequently mistranscribed. The lexicon can include additional metadata (e.g., English glosses) if needed.
2. **Phonetic conversion:** each entity is converted into a sequence of IPA-like syllables with tone markers. The same conversion is applied to sliding windows of the ASR transcript, avoiding brittle word segmentation.
3. **Similarity scoring:** the weighted feature edit distance from `panphon` measures the phonetic gap between the ASR substring and the lexicon entry. Distances are normalised by syllable count to stay comparable across different lengths.
4. **Correction:** candidate corrections are ranked by similarity, and the best non-overlapping corrections are applied greedily. You can adjust the threshold or replace the greedy logic with a more advanced resolver if desired.

## Testing

The repository uses `pytest`. After installing the project with the development extras, run:

```bash
pytest
```

## Limitations and next steps

- The IPA mapping is approximate and prioritises relative similarity over strict phonetic accuracy. Feel free to customise the tables in `phonetics.py` for your data.
- Tone handling is coarse; if tone is unreliable in your ASR output you may want to down-weight it or ignore it altogether.
- For large lexicons, consider caching IPA conversions or pruning candidates by length/frequency to keep runtime manageable.
