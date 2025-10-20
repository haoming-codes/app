# ASR Correction Toolkit

Utilities for correcting Chinese/English multilingual ASR transcripts using a knowledge-base-driven, retrieval augmented (RAG-like) approach. The toolkit compares ASR substrings against a curated entity list using phonetic and tonal distance metrics to surface likely corrections.

## Installation

The project ships as a standard Python package. Install it in editable mode for experimentation:

```bash
pip install -e .[development]
```

This command installs the runtime dependencies (phonemizer with `espeak-ng`, PanPhon, CLTS, the custom `abydos` fork) as well as development extras such as `pytest`.

## Usage

```python
from asr_correction import (
    Corrector,
    DistanceComputationConfig,
    SegmentalMetricConfig,
    ToneDistanceConfig,
    compute_distance,
)

knowledge_base = [
    "阿里巴巴",
    "DeepMind",
    "AG.AL",  # Acronym entries are treated as letter sequences automatically
]

segmental_metrics = [
    SegmentalMetricConfig("panphon", weight=0.6),
    SegmentalMetricConfig("phonetic_edit_distance", weight=0.3),
    SegmentalMetricConfig("aline", weight=0.1),
]

tone_config = ToneDistanceConfig(
    weight=1.0,
    confusion_penalties={
        ("2", "3"): 0.25,
        ("3", "4"): 0.4,
    },
    insertion_penalty=0.8,
    deletion_penalty=0.8,
)

config = DistanceComputationConfig(
    segmental_metrics=segmental_metrics,
    tone=tone_config,
    tradeoff_lambda=0.7,
    threshold=0.45,
)

corrector = Corrector(knowledge_base, config=config)

asr_transcript = "今天我们讨论了阿里爸爸的最新研究，特别是AGAL平台的进展。"

# Query the corrector for suggestions under the configured distance threshold
suggestions = corrector.suggest(asr_transcript)
for suggestion in suggestions:
    print(suggestion)

# Apply the best non-overlapping corrections to the original transcript
corrected = corrector.apply_best(asr_transcript)
print(corrected)

# During hyper-parameter tuning you may want to inspect raw distances
candidate = "阿里爸爸"
reference = "阿里巴巴"
raw_distance = compute_distance(candidate, reference, config)
print("Distance:", raw_distance)
```

### Hyperparameters

* **Segmental metrics** – specify one or more IPA-based metrics via `SegmentalMetricConfig`, each with individual weights and optional constructor arguments.
* **Tone distance** – configure tone penalties through `ToneDistanceConfig`, including per-tone confusion costs and insertion/deletion penalties.
* **Trade-off lambda** – `tradeoff_lambda` interpolates between segmental (1.0) and tonal (0.0) emphasis. Combine with the tone weight to capture your desired balance.
* **Threshold** – `threshold` limits which knowledge-base candidates are surfaced as correction suggestions.

All configuration objects are plain dataclasses and thus straightforward to serialize for experiment tracking.

## Testing

Run the automated test suite after installing the development extras:

```bash
pytest
```

The tests cover distance calculations, tone-aware comparisons, and end-to-end correction suggestions.
