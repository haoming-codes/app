# Phonetic Corrector

Utilities for correcting Chinese ASR named-entity errors with phonetic
matching. The package implements a retrieval-augmented correction pipeline:

1. Convert the ASR output and a curated list of named entities to IPA using
   `pypinyin`'s IPA backend.
2. Compare every contiguous span in the transcription against the entity list
   with a phonetic distance metric. `panphon` is used when available and the
   library gracefully falls back to a SequenceMatcher ratio when it is not.
3. Apply the highest scoring, non-overlapping matches whose similarity is above
   a configurable threshold.

This avoids brittle word segmentation and focuses on the phonetic footprint of
entities that the ASR system frequently mistranscribes.

## Installation

```bash
pip install -e .
```

The package depends on `pypinyin` (for IPA conversion) and `panphon` (for
feature-based phonetic distance). Install-time dependency resolution will fetch
both automatically.

## Usage

```python
from phonetic_corrector import ASRNamedEntityCorrector

entities = [
    "阿里巴巴",
    "马云",
    "张三",
]
corrector = ASRNamedEntityCorrector(entities, threshold=0.65)

transcription = "阿里爸爸的创始人是马允"
corrected, details = corrector.correct(transcription)

print(corrected)
for item in details:
    print(item)
```

Running the snippet above would output the corrected transcription and the
applied replacements with their phonetic similarity scores.

## Development

The project uses a standard setuptools configuration in ``pyproject.toml`` so
editable installs work out of the box. After installing the dependencies you can
run unit tests with `pytest` once you have added test cases.
