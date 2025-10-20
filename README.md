# ASR Corrector

This repository provides utilities for post-processing Mandarin ASR output when an
acoustic model misrecognises named entities. It implements a RAG-inspired matching
pipeline that searches over all substrings of the ASR transcript and replaces them
with canonical entities according to phonetic similarity.

## How it works

1. **Phonetic encoding** – Each candidate entity and each substring from the ASR
   output is converted into an IPA-like representation. By default you can rely on
   the optional `PinyinPanphonEncoder`, which converts characters to pinyin using
   `pypinyin` and then to IPA symbols using its IPA utilities. Any custom encoder
   implementing :class:`asr_corrector.phonetics.BasePhoneticEncoder` can be used.
2. **Similarity scoring** – Phonetic strings are compared with a distance metric.
   The library ships with :class:`asr_corrector.matcher.PanphonDistance`, which
   wraps PanPhon's feature edit distance, and a pure-Python Levenshtein fallback
   used for testing.
3. **RAG-like retrieval** – The :class:`asr_corrector.corrector.NameCorrectionPipeline`
   scans every substring up to the maximum entity length, finds the most similar
   entry in the lexicon, and keeps high-confidence matches without performing
   explicit word segmentation.

## Installation

The project is packaged with `pyproject.toml` so it can be installed in editable
mode while you iterate:

```bash
pip install -e .
```

To enable IPA conversion based on `pypinyin` and PanPhon you can install the
optional dependencies:

```bash
pip install -e .[phonetics]
```

## Usage example

```python
from asr_corrector.corrector import NameCorrectionPipeline
from asr_corrector.entities import EntityLexicon
from asr_corrector.matcher import PanphonDistance
from asr_corrector.phonetics import PinyinPanphonEncoder

lexicon = EntityLexicon.from_surfaces([
    "马斯克",
    "张三丰",
], encoder=PinyinPanphonEncoder())

pipeline = NameCorrectionPipeline(
    lexicon=lexicon,
    encoder=PinyinPanphonEncoder(),
    distance_metric=PanphonDistance(),
    distance_threshold=2.5,
)

asr_text = "今天我们采访了马斯科和张三峰"
result = pipeline.correct(asr_text)
print(result.corrected_text)  # -> 今天我们采访了马斯克和张三丰
```

When optional dependencies are unavailable (for example offline environments) you
can still plug in your own encoder and distance metric implementations.
