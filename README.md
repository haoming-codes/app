# ASR Corrector

Utilities for correcting Chinese ASR outputs that contain common name-entity transcription errors. The package builds a simple retrieval-like index of known entities and matches the ASR output against them using phonetic similarity derived from IPA projections.

## Features

- Convert Mandarin text to IPA with tone markers using a lightweight `pypinyin`-based transcriber.
- Compute similarity scores using `panphon` (when available) or fall back to a string-based ratio.
- Build name lexicons with canonical entities and aliases, ready for retrieval.
- Apply RAG-style correction passes over ASR output strings.

## Installation

The project is distributed as a standard Python package. Install it in editable mode for development:

```bash
pip install -e .
```

Optionally install the phonetic feature dependency for higher quality similarity metrics:

```bash
pip install -e .[phonetics]
```

## Usage

```python
from asr_corrector import NameLexicon, RagBasedCorrector

records = [
    {"canonical": "张伟", "aliases": ["章惟"]},
    {"canonical": "王芳", "aliases": ["王方"]},
]
lexicon = NameLexicon.from_records(records)
corrector = RagBasedCorrector(lexicon, threshold=0.5)
text = "今天我遇到了章惟和王方"
result = corrector.apply(text)
print(result.text)
# 今天我遇到了张伟和王芳
```

## Tests

Run the test suite with:

```bash
PYTHONPATH=src pytest
```

`panphon` is optional. If it is not installed the library gracefully falls back to a sequence-based similarity score.
