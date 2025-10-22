# Multilingual ASR correction toolkit

This project implements a retrieval-augmented approach for correcting
multilingual (Chinese/English) automatic speech recognition outputs with a
knowledge base of named entities and domain-specific jargon. The core idea is
simple: convert both the ASR hypothesis and the knowledge base entries to a
common phonetic representation, measure how close they are, and propose
corrections whenever the distance falls below a configurable threshold.

The repository exposes a small Python package, :mod:`asr_correction`, that can
be installed in editable mode and imported into existing pipelines.

## Installation

```bash
pip install -e .
```

The package depends on [`phonemizer`](https://github.com/bootphon/phonemizer),
[`panphon`](https://github.com/dmort27/panphon), [`pypinyin`](https://github.com/mozillazg/python-pinyin),
and the fork of [`abydos`](https://github.com/denizberkin/abydos) specified in
the project requirements. Make sure the `espeak-ng` binary is available on your
system so that the phonemizer backend can produce IPA output.

## Algorithm overview

1. **Grapheme-to-phoneme conversion** – Chinese segments are mapped to Pinyin
   using ``pypinyin`` and then to IPA using an explicit initial/final inventory
   augmented with tone numbers. English segments are phonemized with the
   ``espeak`` backend from ``phonemizer`` with stress marks preserved.
2. **Feature representation** – Each IPA segment is converted to a PanPhon
   articulatory feature vector (values in \{-1, 0, +1\}). Suprasegmental
   annotations (Mandarin tones and English stress) are attached to the
   syllabic nuclei in the sequence.
3. **Segment distance** – A configurable phonetic string metric is applied to
   the concatenated IPA strings. Two options are provided:
   :class:`abydos.distance.PhoneticEditDistance` and
   :class:`abydos.distance.ALINE`.
4. **Feature alignment** – The feature-vector sequences are aligned with
   Dynamic Time Warping (DTW). The local cost can be an L1, L2 or cosine
   distance over the feature vectors. The resulting average path cost is
   normalized to \[0, 1\].
5. **Suprasegmental penalties** – Tone and stress mismatches are tallied along
   the DTW alignment path. Each mismatch contributes a configurable penalty and
   the totals are normalized by the number of aligned tone/stress pairs.
6. **Weighted combination** – The three components (segment distance, feature
   alignment cost, tone/stress penalties) are combined using the weights
   (lambdas) supplied in :class:`asr_correction.config.DistanceConfig`.
7. **Sliding-window matching** – Given a knowledge base of canonical forms, the
   matcher slides windows over the ASR hypothesis, computes the combined
   distance against each knowledge base entry, and reports candidates below the
   configured threshold.

All steps are language-aware but do not rely on orthographic similarity; even
all-cap entries (e.g. ``AG.AL``) are automatically expanded into letter-by-
letter pronunciations.

## Usage

```python
from asr_correction import DistanceCalculator, DistanceConfig, IPAConverter
from asr_correction import KnowledgeBaseMatcher

# Configure the distance metric.
config = DistanceConfig(
    segment_metric="phonetic_edit",
    feature_distance="l1",
    lambda_segment=0.35,
    lambda_features=0.45,
    lambda_tone=0.15,
    lambda_stress=0.05,
    tone_penalty=1.0,
    stress_penalty=0.5,
    threshold=0.30,
    min_window_size=1,
    max_window_size=4,
)

converter = IPAConverter()
calculator = DistanceCalculator(config=config, converter=converter)

# Compute IPA representations to tune hyperparameters.
ipa_left = converter.ipa("mango")
ipa_right = converter.ipa("美国")
print("IPA representations:", ipa_left, ipa_right)

# Inspect the detailed distance breakdown between two substrings.
breakdown = calculator.distance("mango", "美国")
print(
    "Segment={:.3f} Feature={:.3f} Tone={:.3f} Stress={:.3f} Overall={:.3f}".format(
        breakdown.segment,
        breakdown.features,
        breakdown.tone,
        breakdown.stress,
        breakdown.overall,
    )
)

# Run the knowledge base matcher on an ASR hypothesis.
knowledge_base = ["美国", "成哥", "Mini Map", "AG.AL"]
matcher = KnowledgeBaseMatcher(knowledge_base, config=config, converter=converter)

transcription = "我们讨论了mango, 然后提到了Mini Map"
for match in matcher.find_matches(transcription, threshold=0.3):
    print(
        f"Window={match.window!r} -> {match.entry.original!r} score={match.breakdown.overall:.3f}"
    )
```

The configuration object exposes the most important knobs:

- **``threshold``** – overall distance required for a match.
- **``lambda_*`` weights** – trade-off between segment edit distance, feature
  alignment cost, and suprasegmental penalties.
- **``feature_distance``** – choose between ``"l1"``, ``"l2"`` and
  ``"cosine"`` for the DTW local cost.
- **``tone_penalty`` / ``stress_penalty``** – contribution of tone/stress
  mismatches. Setting these to zero ignores suprasegmental cues.
- **``min_window_size`` / ``max_window_size``** – number of tokens combined when
  extracting candidate windows from the ASR output.

Because :class:`asr_correction.distance.DistanceCalculator` returns a
:class:`~asr_correction.distance.DistanceBreakdown` object, downstream systems
can inspect individual components to choose the most appropriate correction
strategy.

## Testing

Run the unit test suite with:

```bash
pytest
```

The tests cover IPA conversion, distance calculations under different
hyperparameters, and end-to-end knowledge base matching.
