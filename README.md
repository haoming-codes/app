# ASR Corrector

Tools for correcting multilingual Chinese/English ASR outputs using phonetic and tonal distance metrics and a retrieval-augmented knowledge base of entities.

## Installation

Install the project in editable mode to simplify experimentation:

```bash
pip install -e .
```

This package depends on a fork of `abydos`:

```bash
pip install git+https://github.com/denizberkin/abydos.git
```

## Usage

```python
from asr_corrector import (
    Corrector,
    DistanceCalculator,
    DistanceConfig,
    KnowledgeBase,
    KnowledgeEntry,
    SegmentalConfig,
    SegmentalMetric,
    ToneConfig,
    ToneMetric,
)

# Configure distance metrics and weights.
config = DistanceConfig(
    segmental=SegmentalConfig(
        metric=SegmentalMetric.ABYDOS_PHONETIC,
        clts_vector_distance="euclidean",
    ),
    tone=ToneConfig(
        metric=ToneMetric.CONFUSION,
        confusion_costs={
            "1": {"2": 0.5, "3": 0.7, "4": 1.2, "5": 0.3},
            "2": {"1": 0.5, "3": 0.6, "4": 0.8, "5": 0.4},
            "3": {"1": 0.7, "2": 0.6, "4": 0.7, "5": 0.8},
            "4": {"1": 1.2, "2": 0.8, "3": 0.7, "5": 0.9},
            "5": {"1": 0.3, "2": 0.4, "3": 0.8, "4": 0.9},
        },
        default_cost=1.5,
    ),
    segmental_weight=1.0,
    tone_weight=0.4,
    threshold=3.0,
)

# Build a knowledge base of canonical entities.
kb = KnowledgeBase(
    [
        KnowledgeEntry(canonical="阿里巴巴", language="cmn-Hans"),
        KnowledgeEntry(canonical="AG.AL", language="eng-Latn"),
        KnowledgeEntry(canonical="quantum annealing", language="eng-Latn"),
    ]
)

# Quickly experiment with distances between strings for hyper-parameter tuning.
calculator = DistanceCalculator(config)
distance = calculator.distance(
    source="阿里巴巴", target="阿里爸爸", source_language="cmn-Hans"
)
print("distance", distance)

# Inspect individual components when tuning.
details = calculator.explain(
    source="quantum eniling",
    target="quantum annealing",
    source_language="eng-Latn",
)
print(details)

# Run the corrector over an ASR transcript.
corrector = Corrector(kb, config)
transcript = "我们今天讨论的是quantum eniling的最新进展"
print(corrector.apply_best(transcript))
```

## Testing

Run unit tests with pytest:

```bash
pytest
```
