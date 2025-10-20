# ASR Corrector

Utilities for applying retrieval-augmented, phonetics-aware post-corrections to Chinese ASR transcripts. The project focuses on spotting likely errors involving proper nouns and jargon by comparing substrings in an automatic transcription against a domain lexicon using phonetic similarity metrics.

## Approach

1. **Syllabification and phonetic projection** – The `PhoneticConverter` maps Chinese text to Mandarin syllables (via `pypinyin`), converts each syllable into an IPA string using CLTS (with graceful fallbacks), and extracts tone numbers separately.
2. **Segmental similarity** – IPA sequences without tone marks are scored using pluggable segmental metrics. PanPhon's feature edit distance is the default, but wrappers for Abydos' Phonetic Edit Distance and ALINE as well as a simple Levenshtein fallback are provided.
3. **Tone similarity** – Tones are compared through a weighted edit distance (`ToneDistance`). Substitution costs encode common ASR confusions (e.g., Tone 2 ↔ Tone 3), and insertions/deletions model skipped or hallucinated tones.
4. **Combination and filtering** – Segmental and tonal distances are combined with configurable trade-off weights. Candidates below a configurable threshold are exposed for downstream correction or human review. The `PhoneticLexiconMatcher` also offers a helper to apply non-overlapping replacements greedily.

This mirrors a lightweight RAG-like loop: a curated lexicon provides the "retrieved" targets, while phonetic matching replaces the expensive embedding-based similarity step.

## Installation

```bash
pip install -e .
```

The package requires Python 3.9+ and depends on `pypinyin`, `panphon`, `abydos`, and `pyclts`. Optionally, install `lingpy` to experiment with additional CLTS-backed systems.

## Usage

```python
from asr_corrector import (
    LexiconEntry,
    MatcherConfig,
    PanphonFeatureMetric,
    PhoneticConverter,
    PhoneticLexiconMatcher,
    ToneDistance,
)
from asr_corrector.metrics import AbydosALINEMetric

lexicon = [
    LexiconEntry("华为"),
    LexiconEntry("麒麟9000"),
    LexiconEntry("超导量子比特"),
]

converter = PhoneticConverter(neutral_tone_with_five=True)
segment_metric = PanphonFeatureMetric()  # try AbydosALINEMetric() for ALINE

tone_metric = ToneDistance(
    substitution_costs={(2, 3): 0.4, (3, 2): 0.4},  # tighten a specific confusion
    deletion_cost=1.2,
    insertion_cost=1.2,
)

config = MatcherConfig(
    segment_metric=segment_metric,
    tone_metric=tone_metric,
    lambda_segment=0.75,
    lambda_tone=0.35,
    distance_threshold=0.9,
)

matcher = PhoneticLexiconMatcher(lexicon, converter=converter, config=config)

transcript = "我们测试话为手机的最新功能"

matches = matcher.find_matches(transcript)
for match in matches:
    print(
        match.observed,
        "→",
        match.replacement,
        f"score={match.total_distance:.3f}",
        f"segment={match.segment_distance:.3f}",
        f"tone={match.tone_distance:.3f}",
    )

corrected = matcher.apply_best_corrections(transcript, matches)
print("Corrected:", corrected)
```

Adjust the hyperparameters (`lambda_segment`, `lambda_tone`, `distance_threshold`) and swap in different segmental metrics depending on your error tolerance and lexicon size. You can also pre-filter candidates by lexicon length or extend `ToneDistance` with a richer confusion matrix derived from your ASR confusion statistics.

## Roadmap

- Cache prepared lexicon pronunciations to disk for large vocabularies.
- Add optional fuzzy search on Romanised forms to reduce the search space for long transcripts.
- Provide utilities for integrating with ASR post-processing pipelines (streaming or batch).
