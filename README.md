# rag-asr-correction

Utilities for correcting multilingual (Mandarin Chinese and English) ASR transcriptions with
knowledge-base lookups and phonetic distances. The approach assumes API access to the ASR system
and focuses on post-processing the textual transcription.

## Approach

1. **Knowledge base pre-processing**
   - Canonical entity/jargon forms are phonemized with `espeak-ng` via `phonemizer`.
   - Mandarin tone marks are stripped from the IPA strings and handled separately.
   - Tone sequences are extracted with `pypinyin` (numeric tone representation).
   - Uppercase entries are treated as acronyms and expanded into spelled-out letters.

2. **Distance computation**
   - Segmental distance can be any weighted combination of PanPhon weighted feature edit distance,
     Abydos phonetic edit distance, and Abydos ALINE.
   - Tone distance is computed with a weighted edit distance over tone sequences. Separate insertion,
     deletion, and substitution costs let you encode tone confusion tables.
   - The overall score is `sum(metric_weight * metric_distance) + tradeoff_lambda * tone_weight * tone_distance`.

3. **Correction**
   - The ASR output is tokenised into Chinese character runs, Latin words, and punctuation.
   - Sliding windows (bounded by the knowledge-base token length) are compared against all entries.
   - Corrections whose combined score is below the configurable threshold are accepted greedily from
     left to right.

This design allows you to integrate jargon- or entity-specific knowledge without retraining or
fine-tuning the ASR model while keeping Mandarin tone handling explicit.

## Installation

The project uses a standard `pyproject.toml` layout and supports editable installs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

The dependency on [`abydos`](https://github.com/denizberkin/abydos) is pulled from the specified
GitHub fork automatically.

## Usage

```python
from rag_asr_correction import (
    CorrectionConfig,
    CorrectionEngine,
    KnowledgeBase,
    SegmentalMetric,
    SegmentalMetricConfig,
    ToneDistanceConfig,
    DistanceComputer,
)

# Configure the phonetic metrics and thresholds.
config = CorrectionConfig(
    segmental_metrics=[
        SegmentalMetricConfig(SegmentalMetric.PANPHON_WEIGHTED, weight=1.0),
        SegmentalMetricConfig(SegmentalMetric.ABYDOS_ALINE, weight=0.5),
    ],
    tone_config=ToneDistanceConfig(
        weight=1.0,
        substitution_costs={
            (1, 2): 0.35,  # Tone 1 misheard as tone 2 is relatively cheap.
            (3, 4): 0.75,
        },
        insertion_cost=0.6,
        deletion_cost=0.6,
    ),
    threshold=4.5,
    tradeoff_lambda=0.8,  # Balances tone distance with segmental distance.
    max_window_size=5,
)

knowledge_base = KnowledgeBase([
    "张伟",
    "AG.AL",
    "AlphaFold",
])
engine = CorrectionEngine(knowledge_base, config)

transcript = "今天我们采访了章伟博士和AGAL团队, 讨论alphafold的突破。"
result = engine.correct(transcript)
print(result.corrected)
# -> 今天我们采访了张伟博士和AG.AL团队, 讨论AlphaFold的突破。

# Inspect corrections (scores, tone/segmental breakdown).
for candidate in result.applied:
    print(candidate.window_text, "->", candidate.entry.text, candidate.score)

# Hyperparameter tuning helper: compute distance directly.
computer = DistanceComputer(config)
print(
    computer.distance_between_strings("章伟", "张伟", lang_a="cmn", lang_b="cmn")
)
print(
    computer.distance_between_strings("ag al", "AG.AL")
)
```

Adjust the `threshold`, `tradeoff_lambda`, metric selection, and tone costs to match your empirical
error distribution.

## Testing

Run the automated tests with `pytest`:

```bash
pytest
```

The tests exercise the distance interface, tone penalties, and the end-to-end correction flow.
