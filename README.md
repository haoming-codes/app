# ASR Named Entity Corrector

`asr-ne-corrector` provides utilities for post-processing Chinese ASR
transcripts. It focuses on fixing named-entity transcription errors by matching
candidate substrings against a curated list of entities using IPA-based
similarity derived from `pypinyin` and `panphon`.

## Installation

```bash
pip install -e .
```

The project depends on [`pypinyin`](https://github.com/mozillazg/python-pinyin)
and [`panphon`](https://github.com/dmort27/panphon). Both packages are declared
in `pyproject.toml` and will be installed automatically.

## Usage

```python
from asr_ne_corrector import NameEntityCorrector

entities = [
    "上海",
    "西安交通大学",
    "周杰伦",
]

corrector = NameEntityCorrector(entities, similarity_threshold=0.65)

document = "我在上海交大听到周接轮的演唱会"
corrected, matches = corrector.correct(document)

print(corrected)
# -> "我在西安交通大学听到周杰伦的演唱会"

for match in matches:
    print(match.observed, "->", match.replacement(), match.score)
```

### How it works

1. **Phonetic encoding** – Both the observed substring and the canonical
   entities are converted to IPA with `pypinyin`.
2. **Feature-based similarity** – `panphon` computes the normalized feature edit
   distance between the IPA strings. A similarity score of `1 - distance` is
   used to rank candidate replacements.
3. **Span selection** – High-confidence matches are greedily chosen so that no
   spans overlap, and their canonical forms replace the ASR output.

Adjust the similarity threshold and the length tolerance to balance recall and
precision for your particular ASR system and entity list.
