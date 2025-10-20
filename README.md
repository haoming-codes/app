# ASR Corrector

This repository provides a light-weight, configurable toolkit for correcting
Chinese/English multilingual ASR transcriptions. It combines a small
knowledge-base of problematic named entities or jargons with phonetic matching
based on IPA representations. The matching strategy is inspired by RAG-style
retrieval: we slide windows over the ASR output, compute phonetic distances
between each window and the knowledge base entries, and keep the closest
matches when they fall below a configurable threshold.

## Key features

- IPA conversion powered by [Epitran](https://github.com/dmort27/epitran).
- Segmental distances with PanPhon, ALINE, CLTS features, and the custom fork of
  `abydos` that exposes `PhoneticEditDistance`.
- Mandarin tone scoring handled separately from segmental distances with a
  confusion-weighted table.
- Sliding-window retrieval that can be tuned with window sizes, thresholds, and
  trade-off coefficients.
- An ergonomic `ASRCorrector` class that both proposes corrections and exposes a
  distance API for hyperparameter tuning.

## Installation

```bash
pip install -e .
```

The project depends on a fork of `abydos`, installed directly from GitHub, so we
recommend using a virtual environment.

## Usage

Below is a complete example showing how to configure the matcher, register a
small knowledge base, and request correction suggestions. The same configuration
object can be reused to probe distances between any two substrings to assist
hyperparameter search.

```python
from asr_corrector import (
    ASRCorrector,
    DistanceConfig,
    FeatureDistance,
    KnowledgeBase,
    KnowledgeBaseEntry,
    MatcherConfig,
    SegmentalMetric,
    SegmentalMetricConfig,
    default_tone_confusions,
)

# Construct a knowledge base. Each entry records the canonical surface form and
# its language. You can mix English and Mandarin freely.
entries = [
    KnowledgeBaseEntry("史蒂芬·库里", language="zh"),
    KnowledgeBaseEntry("Transformer", language="en"),
    KnowledgeBaseEntry("量子纠缠", language="zh"),
]
knowledge_base = KnowledgeBase(entries)

# Configure the phonetic distance. Combine multiple segmental metrics and adjust
# their weights. Tone scoring is activated by giving a positive weight.
distance_config = DistanceConfig(
    segmental_metrics=[
        SegmentalMetricConfig(SegmentalMetric.PANPHON, weight=0.6),
        SegmentalMetricConfig(SegmentalMetric.PHONETIC_EDIT, weight=0.3),
        SegmentalMetricConfig(
            SegmentalMetric.CLTS,
            weight=0.1,
            feature_distance=FeatureDistance.COSINE,
        ),
    ],
    tone_weight=0.5,
    tone_confusion=default_tone_confusions(),
    segmental_scale=1.0,
)

# Configure the matcher: window sizes, maximum candidates, and acceptance
# threshold (after multiplying by `tradeoff_lambda`).
matcher_config = MatcherConfig(
    window_sizes=(1, 2, 3, 4),
    distance_threshold=1.8,
    tradeoff_lambda=1.0,
    max_candidates=5,
)

corrector = ASRCorrector(knowledge_base, matcher_config, distance_config)

asr_text = "今天我们讨论的是transformer模型和量子纠缠"
result = corrector.suggest(asr_text)

for candidate in result.candidates:
    print(candidate.substring, "->", candidate.entry.surface, candidate.distance.total)

# --- Hyperparameter tuning helper ---
# Directly inspect the breakdown of distances between two strings under the
# current configuration.
breakdown = corrector.compute_distance("梁子纠缠", "量子纠缠", language="zh")
print(breakdown.total, breakdown.details)
```

## Repository layout

```
asr_corrector/
    __init__.py
    config.py
    distance.py
    knowledge_base.py
    matcher.py
    pipeline.py
    tones.py
pyproject.toml
README.md
```

Each module contains docstrings describing the responsibilities and expected
behaviour. The code favours readability and explicitness so that you can extend
or integrate it into your own pipelines.
