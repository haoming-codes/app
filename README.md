# ASR Post-Processor

A lightweight toolkit for correcting Mandarin Chinese ASR transcriptions that mis-handle
named entities or jargon. The library converts both ASR output and a curated lexicon of
canonical terms into IPA + tone sequences, computes a configurable segmental distance,
and applies replacements when the ASR hypothesis is within a threshold.

## Features

- Rule-based conversion from Hanzi to Pinyin and IPA with tone extraction.
- Segmental distances backed by PanPhon, ALINE, or Abydos' Phonetic Edit Distance.
- Separate tone-distance module with a configurable confusion matrix.
- Greedy matching that scans substrings around the target length to recover misheard entities.
- Simple API for correcting single strings or batches of transcripts.

## Installation

The project uses a standard `pyproject.toml`. Install it in editable mode during
experimentation:

```bash
pip install -e .
```

## Usage

```python
from asr_postprocessor import (
    CandidateTerm,
    ChineseASRCorrector,
    CombinedDistanceConfig,
    MatcherConfig,
    SegmentalDistanceConfig,
    ToneDistanceConfig,
)

lexicon = [
    CandidateTerm("阿里巴巴"),
    CandidateTerm("华为"),
    CandidateTerm("高通"),
]

segmental_cfg = SegmentalDistanceConfig(
    metric="panphon",
    method="weighted_feature_edit_distance_div_maxlen",
)

# A softer tone penalty where the neutral tone is often confused.
tone_cfg = ToneDistanceConfig(
    confusion_matrix=[[0.0, 0.5, 0.7, 0.8, 0.4],
                      [0.5, 0.0, 0.4, 0.7, 0.5],
                      [0.7, 0.4, 0.0, 0.5, 0.4],
                      [0.8, 0.7, 0.5, 0.0, 0.6],
                      [0.4, 0.5, 0.4, 0.6, 0.0]],
    insertion_penalty=0.5,
)

combined_cfg = CombinedDistanceConfig(
    segmental=segmental_cfg,
    tone=tone_cfg,
    segment_weight=1.0,
    tone_weight=0.25,
)

matcher_cfg = MatcherConfig(
    max_window_expansion=2,
    distance=combined_cfg,
    threshold=0.48,
)

corrector = ChineseASRCorrector(lexicon, matcher_config=matcher_cfg)

hypothesis = "今天我们访问了阿里八八总部并与华为高层会面"
result = corrector.correct(hypothesis)

print(result.corrected)
for match in result.matches:
    print(match.candidate.surface, match.distance)
```

### Choosing distance metrics

- `SegmentalDistanceConfig.metric` accepts `"panphon"`, `"aline"`, or `"phonetic_edit"`.
- When using PanPhon you may swap the feature set (`segmental_cfg.panphon_feature_set`) or
  call any other PanPhon distance method through `SegmentalDistanceConfig.method`.
- The tone weight (`CombinedDistanceConfig.tone_weight`) trades off tone vs segment errors.
- `MatcherConfig.threshold` controls how aggressively replacements are applied.
- `MatcherConfig.max_window_expansion` allows the matcher to test substrings whose length is
  within the specified delta of the canonical term, which helps capture insertions/deletions.

## Development

The repository contains a small rule-based Pinyin-to-IPA module. If you encounter syllables
that cannot be converted, update `asr_postprocessor/transcription.py` accordingly. The
matching stack is intentionally modular—swap in your own candidate generator or apply more
advanced conflict resolution as needed.

## License

MIT License.
