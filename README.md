# ASR Post-Processing

Utilities for correcting Chinese automatic speech recognition (ASR) transcripts
by matching noisy entity mentions against a curated catalogue of named
entities.  The matching algorithm relies on phonetic similarity derived from
pinyin-to-IPA conversion and :mod:`panphon` feature distances, which allows it
to operate directly on the raw ASR output without requiring word segmentation.

## Installation

The project uses a modern ``pyproject.toml``-based build with ``setuptools``.
To install in editable mode run:

```bash
pip install --upgrade pip
pip install -e .
```

This will fetch the required runtime dependencies:

- [`pypinyin`](https://github.com/mozillazg/python-pinyin) for IPA conversion.
- [`panphon`](https://github.com/dmort27/panphon) for feature-based edit
  distances.

## Usage

```python
from asr_postproc import EntitySpec, PhoneticEntityCorrector

entities = [
    "张三",
    EntitySpec("李四", metadata={"id": 42}),
]

corrector = PhoneticEntityCorrector(entities, similarity_threshold=0.7)
result = corrector.correct("阿里四今天和张伞一起开会")

print(result.corrected_text)
# 阿李四今天和张三一起开会

for match in result.matches:
    print(match.observed, "->", match.replacement(), match.similarity)
```

The corrector searches over all substrings whose length is close to the target
entity's length.  For each candidate substring it computes the IPA sequence via
``pypinyin`` and obtains a PanPhon feature edit distance with the canonical
entity.  The distance is normalised by the length of the entity and converted to
an intuitive similarity score in ``[0, 1]``.  Matches whose similarity exceeds
the configured threshold are accepted, prioritising the highest scoring match
when overlaps occur.

### Why this approach?

- **Segmentation-free:** Chinese ASR errors often break word boundaries.  By
  evaluating every substring directly we avoid cascading mistakes from a word
  segmenter.
- **Phonetic matching:** Computing similarity in IPA feature space focuses on
  the sounds in the transcript, making it robust to homophones and tone errors.
- **Entity catalogue friendly:** You can attach metadata to each
  :class:`EntitySpec` and plug the result back into downstream retrieval or RAG
  pipelines.

### Configuration hints

- Tune ``similarity_threshold`` based on validation transcripts.
- Increase ``max_length_delta`` when insertions/deletions are common in the
  ASR output.
- If you already know frequent erroneous variants of an entity you can store
  them separately in the catalogue to bias the search toward those spans.
