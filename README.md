# ASR Name Corrector

This project provides a lightweight retrieval-and-correction pipeline for fixing
Chinese automatic speech recognition (ASR) name-entity mistakes. The system
uses a lexicon of trusted entities and compares the ASR hypothesis against the
lexicon using Mandarin IPA approximations and PanPhon feature distances. This
keeps the model-agnostic workflow you described: only the decoded text is
required, and no encoder or embedding hooks from the ASR engine are needed.

## Features

- Mandarin syllable to IPA conversion with tone marks and PanPhon distance.
- Lexicon management for entities that are frequently mistranscribed.
- Phonetic retrieval with optional orthographic regularisation to stabilise
  replacements.
- Drop-in correction utility that returns the corrected string together with
  detailed match diagnostics.

## Installation

Install the package in editable mode to iterate on the correction logic:

```bash
pip install -e .
```

## Usage

```python
from asr_name_corrector import EntityLexicon, EntityCorrector

# 1) Build a lexicon from the high-value entities you care about.
lexicon = EntityLexicon.from_strings([
    "张三",
    "李四",
    "王小明",
])

# 2) Configure the corrector. The threshold caps the acceptable combined score,
#    lambda_weight controls how much orthographic distance influences the score,
#    and length_slack lets you consider slightly longer/shorter spans when
#    searching for matches.
corrector = EntityCorrector(
    lexicon,
    threshold=0.38,
    lambda_weight=0.1,
    length_slack=1,
)

hypothesis = "昨天章三和王小名一起开会"
result = corrector.correct(hypothesis)

print(result.corrected_text)
# -> 昨天张三和王小明一起开会

for replacement in result.replacements:
    print(
        replacement.original,
        "->",
        replacement.replacement,
        "score=", replacement.score,
        "phonetic=", replacement.phonetic_distance,
        "orthographic=", replacement.orthographic_distance,
    )
```

## How it works

1. **Syllable transcription** – Incoming text is segmented into Mandarin
   syllables using `pypinyin`. Each syllable is converted to an approximate IPA
   form with tone marks and fed to PanPhon to access articulatory feature
   vectors.
2. **Candidate retrieval** – For each position in the ASR output, the corrector
   compares spans against lexicon entries of similar length, computing a
   weighted distance that emphasises phonetic similarity. A configurable
   `lambda_weight` controls the contribution of raw character edit distance,
   giving you a knob to suppress spurious tone matches.
3. **Replacement** – Segments whose combined score falls below the `threshold`
   are swapped for the canonical lexicon surface form. The correction trace is
   returned so you can audit why each change was made.

Because the search operates purely on decoded text and a curated lexicon, it can
be bolted onto any Chinese ASR system that exposes an API for transcriptions.
