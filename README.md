# RAG ASR Corrector

Utilities for correcting bilingual (Mandarin Chinese/English) ASR transcriptions
with a retrieval-augmented generation (RAG) style workflow. The library exposes
building blocks for computing pronunciation distances, inspecting bilingual IPA
transcriptions, and scanning an ASR hypothesis against a knowledge base of
named entities or jargon.

## Installation

```bash
pip install -e .[test]
```

The package depends on [`phonemizer`](https://github.com/bootphon/phonemizer)
(backed by `espeak`/`espeak-ng`), [`panphon`](https://github.com/dmort27/panphon),
`dtw-python`, and a fork of [`abydos`](https://github.com/denizberkin/abydos).
Ensure that `espeak-ng` binaries are available on your system so that
phonemization succeeds.

## Algorithm overview

Given two bilingual strings (the ASR output window and a knowledge base entry),
we run the following pipeline:

1. **Language-sensitive phonemization** – split the text into Chinese characters
   and English tokens. Each fragment is phonemized with the `phonemizer`
   library using the `espeak-ng` backend (`language="cmn"` for Mandarin,
   `language="en-us"` for English). Acronyms written in ALL CAPS are expanded
   into letter sequences before phonemization.
2. **Phone normalization** – map IPA phones to articulatory feature vectors via
   `panphon.FeatureTable.word_to_vector_list(..., numeric=True)` so that phones
   from both languages live in a comparable space.
3. **Segment distance** – compute multiple phone-level distances:
   - `abydos.distance.PhoneticEditDistance.dist`
   - `abydos.distance.ALINE.dist`
   - Dynamic Time Warping (DTW) over the articulatory feature vectors with an
     Euclidean or Manhattan local cost.
     The three scores are combined with configurable weights and form the
     overall *segment* component.
4. **Suprasegmental penalties** – extract Mandarin tone digits (when present in
   the phonemizer output) and English stress markers (`ˈ`, `ˌ`). Tone and stress
   mismatches are penalized separately. Each penalty is normalized by the number
   of relevant tokens (Chinese characters or English words).
5. **Final score aggregation** – normalize the segment, tone, and stress
   components to `[0, 1]` and combine them with configurable λ weights. The
   resulting total distance is used to rank candidate corrections.

## Usage

### Inspecting IPA output

```python
from rag_asr_corrector import ipa_tokens

tokens = ipa_tokens('我们去了 Mini Map', config=None)
for token in tokens:
    print(token.token.text, token.token.language, token.ipa)
```

### Computing pronunciation distance

```python
from rag_asr_corrector import (
    DistanceConfig,
    DistanceLambdas,
    PronunciationDistance,
    SegmentWeights,
)

config = DistanceConfig(
    segment_weights=SegmentWeights(
        phonetic_edit=0.5,
        aline=0.3,
        feature=0.2,
    ),
    lambdas=DistanceLambdas(
        segment=0.7,
        tone=0.2,
        stress=0.1,
    ),
    dtw_distance="euclidean",
    tone_penalty=1.0,
    stress_penalty=0.5,
)

distance = PronunciationDistance(config)
breakdown = distance.distance('恒哥', '成哥')
print(breakdown.total)
print(breakdown.segment_components)
```

Tune the hyperparameters (`segment_weights`, `lambdas`, penalties, or the DTW
metric) to match your data. All components are normalized to `[0, 1]`, so the
same configuration can be used to define a decision threshold (e.g. `0.3`).

### Suggesting corrections from a knowledge base

```python
from rag_asr_corrector import ASRCorrector, CorrectionConfig, DistanceConfig

kb = ['美国', '成哥', 'Mini Map']
config = CorrectionConfig(
    distance=DistanceConfig(
        segment_weights=SegmentWeights(phonetic_edit=0.5, aline=0.4, feature=0.1),
        lambdas=DistanceLambdas(segment=0.8, tone=0.1, stress=0.1),
        dtw_distance='manhattan',
    ),
    threshold=0.35,
)

corrector = ASRCorrector(kb, config)
suggestions = corrector.suggest('我们刚见到恒哥在 minivan 上讲解 Mini Map')
for suggestion in suggestions:
    print(suggestion.original_text, '→', suggestion.candidate, suggestion.distance.total)
```

The corrector scans the ASR transcript with sliding windows whose language
patterns match the knowledge base entries. Each window is scored with the
configurable distance metric, and candidates below `threshold` are returned in
ascending order of total distance.

## Testing

```bash
pytest
```

## Notes

- Tone digits may not always be produced by `espeak`/`espeak-ng`. When tones are
  missing from both inputs, the tone penalty naturally evaluates to zero.
- The distance calculators and corrector classes are deterministic; you can
  reuse a single `ASRCorrector` instance across requests to amortize the cost of
  phonemizing the knowledge base.
