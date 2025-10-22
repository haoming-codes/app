# asr-corrector

Tools for correcting multilingual ASR (Chinese/English) outputs using phonetic distance heuristics inspired by retrieval-augmented generation (RAG) workflows. The library focuses on matching ASR substrings against a pronunciation knowledge base of named entities and jargon.

## Installation

```bash
pip install -e .
```

The project depends on [`phonemizer`](https://github.com/bootphon/phonemizer), [`panphon`](https://github.com/dmort27/panphon), and a fork of [`abydos`](https://github.com/denizberkin/abydos) for phonetic edit distances. Ensure that `espeak-ng` voices for English (`en-us`) and Mandarin Chinese (`cmn`) are available in your environment.

## Algorithm overview

1. **Phonemization**
   - Split the input string into language-specific chunks (Mandarin Han characters, Latin-script segments, or other fallbacks).
   - Convert each chunk to IPA with `phonemizer` (English via `en-us`, Mandarin via `cmn`). Acronyms are expanded letter by letter before phonemization.
   - Remove spaces and punctuation, retaining suprasegmental markers from `espeak-ng` (stress marks and tone digits when available).
2. **Articulatory features**
   - Map each IPA segment to a PanPhon articulatory feature vector (values in {-1, 0, +1}). Missing segments default to a zero vector.
3. **Segmental distances**
   - Compute a phonetic edit distance between the IPA strings using either `PhoneticEditDistance` or `ALINE` (from `abydos`).
   - Normalize the distance by the length of the aligned phone sequences.
4. **Feature-sequence alignment**
   - Run Dynamic Time Warping (DTW) on the articulatory feature sequences. Local costs can use cosine or Euclidean distance. Normalize by the alignment path length.
5. **Suprasegmental penalties**
   - Extract Mandarin tone digits and English stress marks per syllable. Apply a normalized Levenshtein penalty to each suprasegmental sequence.
6. **Combination**
   - Combine the normalized segmental, feature, tone, and stress distances via weighted averaging. Threshold the combined score to decide whether to flag a match.

## Usage

### Converting text to IPA and feature vectors

```python
from asr_corrector.phonetics import IPAConverter

converter = IPAConverter()
ipa = converter.to_ipa("Mini Map")
print(ipa.ipa)                 # -> "mˈɪnimˈæp"
print(ipa.phones)              # IPA segments
print(ipa.feature_vectors.shape)  # (num_segments, num_features)
```

### Computing distances with configurable hyperparameters

```python
from asr_corrector.config import DistanceConfig
from asr_corrector.distance import DistanceCalculator

config = DistanceConfig(
    segment_metric="phonetic_edit",   # or "aline"
    feature_metric="cosine",          # or "euclidean"
    segment_weight=0.6,
    feature_weight=0.3,
    tone_weight=0.05,
    stress_weight=0.05,
    threshold=0.05,                    # application-specific decision cutoff
)

calculator = DistanceCalculator(config=config)
score = calculator.compute("mango", "美国")
print(score.combined)
print(score.segment, score.feature, score.tone, score.stress)
```

### Matching ASR outputs against a knowledge base

```python
from asr_corrector.matcher import KnowledgeBaseMatcher, KnowledgeBaseEntry
from asr_corrector.config import MatchingConfig, DistanceConfig

kb = [
    KnowledgeBaseEntry("美国"),
    KnowledgeBaseEntry("成哥"),
    KnowledgeBaseEntry("Mini Map"),
]

matching_config = MatchingConfig(
    window_sizes=(1, 2, 3, 4),
    decision_threshold=0.05,
    distance=DistanceConfig(segment_metric="phonetic_edit", feature_metric="cosine"),
    require_tone_match=False,
    require_stress_match=False,
)

matcher = KnowledgeBaseMatcher(kb, config=matching_config)
results = matcher.match("今天mango很好吃还有Mini Map的问题")
for match in results:
    print(match.window_text, "->", match.entry.surface, match.distance.combined)
```

Tune the `decision_threshold`, weights in `DistanceConfig`, and window sizes to balance recall and precision for your ASR error patterns. Acronyms in the knowledge base (all caps such as `AG.AL`) are automatically interpreted as letter-by-letter pronunciations.

## Testing

```bash
pytest
```
