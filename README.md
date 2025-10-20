# Chinese ASR Lexicon Corrector

This repository provides utilities to post-process Mandarin ASR transcripts with
a RAG-style lexicon lookup. It converts Chinese text into IPA, measures
phonetic + tonal distance, and surfaces candidate corrections from a
domain-specific lexicon. Tone similarity is scored separately from the
segmental distance and the two signals are combined with configurable weights.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The package depends on a fork of `abydos` that exposes the latest phonetic
metrics. `pip install -e .` will automatically pull it from
`https://github.com/denizberkin/abydos.git`. If you want to use the CLTS-based
segmental distance you must also configure a local CLTS catalogue, e.g.

```bash
pyclts dataset install clts
```

Refer to the [`pyclts` documentation](https://github.com/cldf/pyclts) for
alternative setup options when working offline.

## Usage

The library is structured around three building blocks:

1. **`PhoneticConverter`** converts Chinese strings into IPA syllables and tone
   sequences.
2. **`DistanceCalculator`** measures a composite distance between two strings.
3. **`LexiconCorrector`** scans ASR output and surfaces candidate matches from a
   lexicon under a configurable distance threshold.

### Configuring distance hyperparameters

```python
from asr_corrector import (
    CorrectionConfig,
    DistanceCalculator,
    DistanceConfig,
    ToneDistanceConfig,
)

# Tone confusion weights follow the "lower is better" convention.
tone_config = ToneDistanceConfig(
    insertion_penalty=1.0,
    confusion_matrix={
        1: {1: 0.0, 2: 0.6, 3: 1.0, 4: 0.9, 5: 0.5},
        2: {1: 0.6, 2: 0.0, 3: 0.7, 4: 1.1, 5: 0.6},
        3: {1: 1.0, 2: 0.7, 3: 0.0, 4: 1.2, 5: 0.7},
        4: {1: 0.9, 2: 1.1, 3: 1.2, 4: 0.0, 5: 0.8},
        5: {1: 0.5, 2: 0.6, 3: 0.7, 4: 0.8, 5: 0.0},
    },
)

distance_config = DistanceConfig(
    segmental_metric="panphon",      # panphon | phonetic_edit | aline | clts
    segmental_weight=0.75,           # λ for the segmental term
    tone_weight=0.45,                # λ for the tone term
    segmental_kwargs={},             # forwarded to the segmental metric
    tone=tone_config,
)

config = CorrectionConfig(
    distance=distance_config,
    threshold=3.5,                   # reject matches above this composite score
    max_length_delta=1,              # allow +/- 1 character when sliding windows
    enable_length_normalization=True,
)

calculator = DistanceCalculator(config.distance)
result = calculator.measure("重庆大学", "重慶大學", normalize=True)
print(result.total, result.segmental, result.tone)
```

`DistanceCalculator.measure()` returns a `DistanceBreakdown` object that exposes
both the total score and the separate segmental and tone contributions. This is
useful when tuning the trade-off between the two components.

### Scanning ASR output with a lexicon

```python
from asr_corrector import CandidateTerm, LexiconCorrector

lexicon = [
    CandidateTerm("阿里巴巴", {"type": "company"}),
    CandidateTerm("埃隆·马斯克", {"type": "person"}),
    "量子纠缠",  # metadata is optional
]

corrector = LexiconCorrector(lexicon, config=config)
asr_hypothesis = "我们邀请阿狸巴巴的代表来讨论量子九缠"

matches = corrector.find_matches(asr_hypothesis)
for match in matches[:3]:
    print(
        match.substring,
        "→",
        match.candidate.surface,
        "score=",
        round(match.score, 3),
        "segmental=",
        round(match.segmental, 3),
        "tone=",
        round(match.tone, 3),
    )
```

The matcher slides a window across the ASR output, evaluates every candidate,
and returns results sorted by the composite distance. The example above allows
windows that are one character shorter or longer than the candidate, enabling
robustness to common insertion/deletion mistakes.

### Computing distances explicitly

For hyperparameter sweeps you can directly call the convenience method
`DistanceCalculator.distance(text_a, text_b, normalize=True)` or reuse the
`DistanceBreakdown` to inspect the per-component scores. This interface is
compatible with any optimisation loop or notebook workflow.

## Notes

- Tone marks are handled independently from the IPA segmental strings so you can
  experiment with different confusion matrices without touching the segmental
  metric.
- The IPA conversion relies on deterministic rules covering the Mandarin
  initial/final inventory. Unknown syllables fall back to the original Pinyin
  string so you can easily spot gaps in the mapping.
- If CLTS resources are unavailable the CLTS-based metric raises a descriptive
  error instructing you to install the dataset.
