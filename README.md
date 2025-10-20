# ASR Post-Correction

This project provides a retrieval-augmented approach for correcting Chinese ASR
(named-entity) transcription errors using IPA-based phonetic similarity.

## Features

- Convert Chinese strings to IPA using [`pypinyin`](https://github.com/mozillazg/python-pinyin).
- Measure phonetic similarity with [`panphon`](https://github.com/dmort27/panphon)
  feature-edit distances.
- Sliding-window search over raw ASR output without pre-segmentation.
- Simple CLI for batch correction and a Python API for integration.

## Installation

```bash
pip install -e .
```

This installs the package in editable mode and exposes the `asr-correct`
command-line tool.

## Usage

### Python API

```python
from asr_postcorrection import CorrectionEngine, LexiconEntry

lexicon = [
    LexiconEntry("王菲"),
    LexiconEntry("张学友"),
]
engine = CorrectionEngine(lexicon, threshold=0.55)
print(engine.correct("我喜欢王非的歌"))
```

### Command Line

Prepare a JSON lexicon:

```json
[
  "王菲",
  {"surface": "张学友", "metadata": {"type": "singer"}}
]
```

Run the CLI:

```bash
asr-correct lexicon.json "我喜欢王非的歌"
```

## Tests

```bash
pytest
```
