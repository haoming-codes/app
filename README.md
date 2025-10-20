# RAG-style Chinese ASR Corrector

This package provides utilities for correcting the output of Chinese ASR
systems using a retrieval-and-generation inspired workflow. Given a list of
named entities (for example, people or locations that your ASR system often
mis-transcribes), the library searches the raw ASR transcript for phonetically
similar substrings and replaces them with the canonical entity form.

The phonetic similarity is computed with IPA representations obtained from
`pypinyin`, while `panphon` provides articulatory feature distances. Tone
mismatches are penalised separately so that tone errors are de-emphasised
relative to segmental errors.

## Installation

```bash
pip install -e .
```

The package requires Python 3.10 or newer.

## Usage

```python
from rag_corrector import Entity, EntityCorrector

entities = [
    Entity(surface="张三"),
    Entity(surface="北京", aliases=["背景"]),
]

corrector = EntityCorrector(entities, score_threshold=0.5)

transcript = "今天章三去了背景旅游"
corrected, corrections = corrector.correct(transcript)

print(corrected)
# 今天张三去了北京旅游

for item in corrections:
    print(item)
```

## Entity data

Entities can be instantiated programmatically or loaded from JSON/JSONL using
`rag_corrector.load_entities`. Each entry should contain a `surface` field and
optional `aliases` for common mis-transcriptions. The `EntityCorrector`
prefers canonical surfaces for replacements even when aliases are matched.

## Testing

Run the unit tests with:

```bash
pytest
```

## Limitations

* The phonetic mapping relies on standard Mandarin pronunciations produced by
  `pypinyin`. Domain-specific pronunciations may require custom overrides.
* Tone distance is incorporated via a lightweight edit-distance heuristic. You
  can adjust the `tone_weight` parameter on the corrector to tune its impact.
