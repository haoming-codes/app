# ASR Corrector

A lightweight RAG-like correction layer for multilingual (Mandarin/English) ASR
transcripts that focuses on repairing named entities and jargon using
phonetic + tonal similarity.

## Installation

The project is packaged so it can be installed in editable mode for rapid
iteration:

```bash
pip install -e .
```

> **Note:** The project depends on eSpeak through `phonemizer`, as well as the
> forked `abydos` package:
>
> ```bash
> pip install git+https://github.com/denizberkin/abydos.git
> ```

## Usage

### Building a knowledge base and correcting transcripts

```python
from asr_corrector import (
    ASRCorrector,
    CorrectionConfig,
    KnowledgeBase,
    KnowledgeBaseEntry,
    SegmentalMetricConfig,
    ToneDistanceConfig,
    default_distance_config,
)

kb = KnowledgeBase([
    KnowledgeBaseEntry(term="张伟"),
    KnowledgeBaseEntry(term="OpenAI"),
    KnowledgeBaseEntry(term="AG.AL"),  # acronym, pronounced A-G-A-L
])

# Customise the distance configuration
custom_distance = default_distance_config()
custom_distance.segmental_metrics = [
    SegmentalMetricConfig(name="panphon_wfed", weight=1.0),
    SegmentalMetricConfig(name="abydos_phonetic_edit", weight=0.8),
    SegmentalMetricConfig(name="abydos_aline", weight=0.6),
]
custom_distance.tone_tradeoff = 0.4
if custom_distance.tone_config:
    custom_distance.tone_config.weight = 1.2

config = CorrectionConfig(
    distance=custom_distance,
    threshold=0.45,
    window_min_tokens=1,
    window_max_tokens=5,
    allow_overlapping_corrections=False,
)

corrector = ASRCorrector(kb, config=config)
transcript = "我们昨天和张薇开会讨论了 AG L 项目"
for suggestion in corrector.suggest(transcript):
    print(suggestion)
```

### Distance-only experiments for hyper-parameter tuning

Use the helper to experiment with different combinations of hyper-parameters
without instantiating the full correction pipeline:

```python
from asr_corrector import compute_distance, default_distance_config

config = default_distance_config()
config.tone_tradeoff = 0.5
score = compute_distance(
    "张薇",
    "张伟",
    distance_config=config,
    language_a="zh",
    language_b="zh",
)
print(f"Combined distance: {score:.3f}")
```

Adjust `config.segmental_metrics`, their `weight`s, and `config.tone_config`
(`default_penalty`, custom confusion tables, etc.) to tune the behaviour. The
`threshold` parameter in `CorrectionConfig` controls when a suggestion is
accepted.

## Development

```bash
pip install -e .[dev]
```

Run the formatting suite before contributing:

```bash
black src
isort src
```

Tests can be executed with `pytest`.
