# ASR Corrector

RAG-inspired correction pipeline for multilingual (Chinese/English) ASR outputs using phonetic distances and a domain knowledge base.

## Installation

```bash
pip install -e .[dev]
```

This installs the package in editable mode together with testing dependencies. The project relies on the custom fork of `abydos` shipped via `pip install git+https://github.com/denizberkin/abydos.git`.

## Usage

```python
from asr_corrector import (
    ASRCorrector,
    CorrectionConfig,
    KnowledgeBase,
    KnowledgeEntry,
    MetricConfig,
    ToneConfig,
)

knowledge_base = KnowledgeBase([
    KnowledgeEntry("AG.AL"),
    KnowledgeEntry("清华大学"),
    KnowledgeEntry("量子纠缠"),
])

config = CorrectionConfig(
    threshold=1.2,
    segment_lambda=0.8,
    tone_lambda=0.2,
    metrics=[
        MetricConfig(name="panphon", weight=0.6),
        MetricConfig(name="phonetic_edit", weight=0.3),
        MetricConfig(name="aline", weight=0.1),
        MetricConfig(name="clts", weight=0.4),
    ],
    tone=ToneConfig(
        weight=0.5,
        confusion_costs={
            "3": {"2": 0.3, "4": 0.6},
            "4": {"2": 0.5},
        },
        default_cost=1.0,
    ),
)

corrector = ASRCorrector(knowledge_base, config)
asr_text = "我们访问了清华大學 并讨论了量子纠缠 在 AGL 项目中"
result = corrector.correct(asr_text)
print(result.text)
for repl in result.replacements:
    print(repl.original, "->", repl.replacement, repl.breakdown.total)
```

### Computing distances for hyper-parameter tuning

Use the `DistanceCalculator` directly to inspect composite distances between candidate substrings and knowledge-base entries.

```python
from asr_corrector import CorrectionConfig, DistanceCalculator, MetricConfig, ToneConfig

config = CorrectionConfig(
    metrics=[MetricConfig(name="panphon", weight=0.5), MetricConfig(name="aline", weight=0.5)],
    tone=ToneConfig(weight=1.0, default_cost=0.8),
    threshold=1.2,
    segment_lambda=0.7,
    tone_lambda=0.3,
)
calculator = DistanceCalculator(config)
breakdown = calculator.distance("清华大學", "清华大学")
print(breakdown.total, breakdown.segmental, breakdown.tone, breakdown.per_metric)
```

### CLI / Programmatic integration

The library is designed for programmatic integration. Instantiate the `ASRCorrector` with your knowledge base and configuration, run it on ASR outputs, and inspect the `CorrectionResult` for applied corrections and their scores.

## Testing

```bash
pytest
```
