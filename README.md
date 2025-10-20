# ASR Phonetic Corrector

This package provides a lightweight post-processing pipeline for Chinese ASR
systems whose output may contain systematic named-entity errors. The approach is
simple: treat the ASR transcript as an unsegmented character sequence and scan
for substrings that are phonetically similar to a set of canonical
named-entities. If a candidate span sounds close enough—measured through
PanPhon's feature edit distance on IPA strings—the substring is replaced by the
canonical entity.

The method avoids Chinese word segmentation entirely. Instead, it enumerates all
substrings with lengths close to the target entity's syllable count; this keeps
coverage high even when the ASR model inserts or deletes characters. Because we
only rely on the published ASR transcript, no decoder/encoder internals are
required, making the approach compatible with API-only ASR access.

## Installation

The project is packaged with a standard `pyproject.toml` so it can be installed
in editable mode during experimentation:

```bash
pip install -e .
```

Dependencies:

- [`pypinyin`](https://github.com/mozillazg/python-pinyin) for robust conversion
  from Hanzi to Pinyin syllables.
- [`panphon`](https://github.com/rtxanson/panphon) for IPA feature distances.

## Usage

```python
from asr_corrector import NameEntity, PhoneticCorrector

entities = [
    NameEntity("阿里巴巴"),
    NameEntity("华为"),
    NameEntity("腾讯"),
]

corrector = PhoneticCorrector(entities, threshold=0.32)
result = corrector.correct("今天我们访问了阿里八八公司的总部")

print(result.corrected)
# 今天我们访问了阿里巴巴公司的总部

for match in result.matches:
    print(match.original, "->", match.entity.canonical, "score=", match.score)
```

### Choosing a threshold

The default `threshold=0.35` works as a conservative starting point. Lower the
threshold to reduce false positives. You can inspect match scores from
`result.matches` to tune the value empirically for your ASR model and
entity list.

### Pre-computing IPA

If you have pre-computed IPA strings for your entities, provide them through the
`NameEntity` constructor (`NameEntity("阿里巴巴", ipa="...")`). The corrector will
reuse the supplied IPA and skip recomputation.

## Extending the entity list

The workflow assumes you maintain a curated list of canonical entity forms.
Whenever you observe a new ASR confusion, simply append the correct entity to
that list. Because matching is phonetic, you do not need to enumerate every
possible spelling error.

## Development

The repository contains minimal infrastructure so you can plug it into your own
pipelines or scripts. Add unit tests under `tests/` and run them with `pytest`
after installing the project in editable mode.
