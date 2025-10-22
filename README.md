# ASR Corrector

Tools for correcting multilingual (Chinese/English) ASR transcripts with a knowledge base of entity and jargon pronunciations. The package implements a configurable pipeline that converts text to IPA, aligns phonetic features, and penalizes tone and stress mismatches to surface likely substitutions.

## Installation

Install the package in editable mode so you can iterate on the configuration easily:

```bash
pip install -e .[test]
```

The project depends on [`phonemizer`](https://github.com/bootphon/phonemizer), [`panphon`](https://github.com/dmort27/panphon), and the [`abydos` fork](https://github.com/denizberkin/abydos) containing the phonetic distance metrics used by the algorithm.

## Algorithm overview

Given a knowledge base of reference pronunciations, the correction pipeline works as follows:

1. **Segmentation and transcription**
   * Heuristically segment the text into Mandarin characters, English words, and acronyms (full-cap sequences are spelled out).
   * Use `phonemizer` with the espeak-ng backend to produce IPA for each segment (`cmn-latn-pinyin` for Mandarin to keep tone numbers, `en-us` with stress marks for English).
   * Strip spacing and punctuation, while storing Mandarin tone digits and English stress symbols.
   * Map the IPA segments to articulatory feature vectors with `panphon`.
2. **Segment distance**
   * Apply one or more string-level phonetic metrics from `abydos` (default: Phonetic Edit Distance and ALINE) to the concatenated IPA strings.
   * Run dynamic time warping (DTW) on the sequences of articulatory feature vectors. The local cost is either L1 or L2 distance between feature vectors.
   * Combine the metric scores and the feature-DTW score with configurable weights to obtain a single segment distance in \[0,1\].
3. **Suprasegmental penalties**
   * Extract the Mandarin tone sequences (tone digits) and English stress sequences (primary/secondary stress markers) from both strings.
   * Compute normalized edit distances on these sequences. Tone penalties are normalized by the longer tone sequence (i.e., number of Mandarin syllables). Stress penalties are normalized by the longer stress sequence.
4. **Final distance**
   * Aggregate segment, tone, and stress scores with user-provided lambdas. The result is a normalized distance where lower is better. This score is used to decide whether an ASR span should be replaced by a knowledge base entry.

A correction is applied when the best-scoring window around an ASR span is below the chosen threshold. The engine avoids overlapping replacements by keeping only the lowest-cost candidates.

## Usage

### Computing IPA and distances

```python
from asr_corrector import (
    DistanceConfig,
    DistanceWeights,
    SegmentDistanceConfig,
    SegmentMetricConfig,
    compute_distance,
    transcribe_to_ipa,
)

ipa = transcribe_to_ipa("美国 Mini Map")
print("IPA:", ipa)

config = DistanceConfig(
    segment=SegmentDistanceConfig(
        metrics=[
            SegmentMetricConfig(name="phonetic_edit_distance", weight=0.7),
            SegmentMetricConfig(name="aline", weight=0.3),
        ],
        feature_weight=1.0,
        metrics_weight=1.0,
        feature_distance="l1",
    ),
    weights=DistanceWeights(segment=0.6, tone=0.3, stress=0.1),
    correction_threshold=0.35,
)

score = compute_distance("mango", "美国", config=config)
print("distance:", score.total)
print("segment:", score.segment, "tone:", score.tone, "stress:", score.stress)
```

### Correcting ASR output

```python
from asr_corrector import CorrectionEngine, DistanceConfig

knowledge_base = [
    "美国",
    "成哥",
    "Mini Map",
]

config = DistanceConfig(
    correction_threshold=0.38,
    window_radius=1,
)

engine = CorrectionEngine(knowledge_base, config=config)
result = engine.correct("他去了mango的minivan上看minimap")
print(result.text)
for cand in result.applied:
    print(cand.original, "->", cand.entry, "score", cand.distance.total)
```

The `window_radius` controls how many extra tokens around a knowledge-base phrase are considered when searching for a match. Setting it to `1` allows the engine to consider windows that are one token shorter or longer than the knowledge phrase.

## Testing

Run the automated tests after adjusting the configuration:

```bash
pytest
```

## License

MIT
