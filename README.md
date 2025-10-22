# Phonetic RAG Corrector

Tools for correcting Chinese/English ASR transcripts by comparing substrings to a knowledge base in a phonetic space. The library exposes utilities to convert text into IPA, compute articulatory feature sequences, and combine multiple phonetic similarity metrics (segmental distance, articulatory DTW, tone, and stress penalties).

## Installation

```bash
pip install --upgrade pip
pip install -e .
```

> [!IMPORTANT]
> The project depends on a fork of `abydos`. Editable installation pulls it automatically because `pyproject.toml` pins the git URL. Ensure `espeak-ng` is available on your system for `phonemizer` to work.

For development extras:

```bash
pip install -e .[test]
```

## Algorithm overview

1. **Tokenize** the input string into Chinese character spans and Latin words/acronyms.
2. **Chinese tokens** are converted to Pinyin with tone numbers (via `pypinyin`) and then phonemized with the Mandarin (`cmn-latn-pinyin`) voice of `espeak-ng`.
3. **English tokens** are phonemized with the American English (`en-us`) voice. Uppercase tokens are treated as acronyms and spelled out letter by letter.
4. **Articulatory features** are extracted using `panphon`, producing ternary vectors for each phone.
5. **Segment distance** between IPA strings uses either ALINE or phonetic edit distance from `abydos`.
6. **Feature DTW** aligns articulatory feature vectors using dynamic time warping with configurable local cost (`L1` or `L2`).
7. **Tone penalties** apply a normalized edit distance on Mandarin tone sequences. **Stress penalties** do the same for English word stresses (primary, secondary, or unstressed).
8. **Score aggregation** linearly combines the normalized metrics with user-specified weights. All components are normalized to [0, 1] before weighting.

This pipeline allows scanning ASR outputs with a sliding window and comparing against knowledge base entries. Low distance scores indicate that the window likely corresponds to the entry, even when the ASR made phonetic mistakes (e.g., "mango" vs. "美国").

## Usage

```python
from asr_corrector import (
    MultilingualPhoneticConverter,
    PhoneticDistanceCalculator,
    PhoneticScoringConfig,
)

# Hyper-parameters for the combined score
config = PhoneticScoringConfig(
    segment_metric="aline",        # or "phonetic_edit"
    segment_metric_kwargs={"normalize": True},
    feature_metric="l1",            # choose between "l1" or "l2"
    segment_weight=0.5,
    dtw_weight=0.3,
    tone_weight=0.1,
    stress_weight=0.1,
)

calculator = PhoneticDistanceCalculator(config=config)

# Distance between two substrings (lower means more similar)
score = calculator.distance("mango", "美国")
print(score)

# Inspect the raw components for tuning
components = calculator.components("minivan", "Mini Map")
print(components)

# Convert text to IPA for manual inspection / hyperparameter tuning
converter = MultilingualPhoneticConverter()
print(converter.ipa("Mini Map 成哥"))
```

### Thresholding example

```python
threshold = 0.35
candidate = "mango"
entry = "美国"
if calculator.distance(candidate, entry) < threshold:
    print("Replace candidate with knowledge base entry")
```

Combine the distance API with sliding window logic around the ASR output to find candidate corrections. Tune the threshold and weights per domain using the provided breakdown and IPA visualizations.

## Testing

```bash
pytest
```
