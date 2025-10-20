# ASR Correction Toolkit

This repository implements a lightweight retrieval-augmented correction pipeline for
multilingual (Chinese/English) ASR transcripts. The focus is on repairing
name-entity and jargon substitutions by comparing ASR windows against a
knowledge base with configurable phonetic distance metrics.

## Installation

```bash
# Install in editable mode so that you can tweak the code and rerun experiments
pip install -e .
```

> **Note**
> This project relies on a fork of `abydos` that exposes the phonetic distance
> implementations we need. The dependency is already declared in
> `pyproject.toml`, but you can install it manually if preferred:
>
> ```bash
> pip install git+https://github.com/denizberkin/abydos.git
> ```

## Usage

### 1. Define your knowledge base

```python
from asr_correction import KnowledgeBase, KnowledgeBaseEntry

kb = KnowledgeBase(
    [
        KnowledgeBaseEntry("阿里巴巴", language="zh"),
        KnowledgeBaseEntry("DeepMind", language="en"),
    ]
)
```

### 2. Configure the distance components

You can mix and match different segmental and tonal metrics. Every component has
an independent weight, and you can inject custom hyperparameters through the
``options`` dictionary.

```python
from asr_correction import CorrectionConfig, DistanceComponentConfig

config = CorrectionConfig(
    threshold=2.0,  # reject suggestions whose weighted distance exceeds this value
    tone_tradeoff=0.7,
    components=[
        DistanceComponentConfig("panphon", weight=0.5),
        DistanceComponentConfig(
            "phonetic_edit",
            weight=0.3,
            options={"costs": {"sub": 1.2, "ins": 1.0, "del": 1.0}},
        ),
        DistanceComponentConfig("aline", weight=0.2, options={"sigma": 0.6}),
        DistanceComponentConfig(
            "tone",
            weight=0.7,
            options={
                "matrix": [
                    [0.0, 1.0, 1.5, 2.0, 2.5],
                    [1.0, 0.0, 1.0, 1.5, 2.0],
                    [1.5, 1.0, 0.0, 1.0, 1.5],
                    [2.0, 1.5, 1.0, 0.0, 1.0],
                    [2.5, 2.0, 1.5, 1.0, 0.0],
                ]
            },
        ),
    ],
)
```

### 3. Run the correction engine

```python
from asr_correction import CorrectionEngine

engine = CorrectionEngine(kb, config)
asr_output = "我们今天访问了阿里爸爸总部并聊到了Deep Mind的研究"
corrected, suggestions = engine.correct(asr_output)

print(corrected)
for suggestion in suggestions:
    print(suggestion)
```

### 4. Hyperparameter tuning helper

To evaluate a specific configuration without running the full correction loop,
use the ``compute_distance`` shortcut. This is handy when sweeping thresholds or
weights.

```python
from asr_correction import compute_distance

distance = compute_distance(
    "阿里爸爸",
    "阿里巴巴",
    language="zh",
    configs=config.components,
    tone_tradeoff=config.tone_tradeoff,
)
print(distance)
```

### 5. Working with English-only entries

The same API works for English segments. Make sure to mark your knowledge-base
entries with ``language="en"`` so that the engine tokenises the ASR output as a
sequence of words.

```python
kb.add(KnowledgeBaseEntry("Transformer", language="en"))
```

## Approach

1. **Transliteration** – Convert Chinese characters to Pinyin and then to IPA
   (with tone numbers stored separately). English terms are converted to IPA via
   `eng_to_ipa`.
2. **Segmental distance** – Compute one or more IPA-based distances (PanPhon
   feature edit, ABYDOS Phonetic Edit Distance, ALINE, CLTS vector distance).
3. **Tone distance** – Compare Mandarin tones with a customisable confusion
   matrix and combine the result with the segmental score.
4. **Windowed matching** – Slide over the ASR transcript using language-specific
   windows to locate the closest matches from the knowledge base.
5. **Correction** – Replace windows whose weighted distance falls below the
   configured threshold.

The components are composable so that you can experiment with different weights
or even drop in your own distance functions.

## Contributing

Feel free to open issues or pull requests if you would like additional features
or better language coverage.
