# ASR Corrector

Utilities for correcting Mandarin Chinese automatic speech recognition (ASR) transcripts by
matching substrings to a jargon/nerd lexicon via phonetic retrieval. The library computes
de-toned IPA transcriptions for both the ASR hypothesis and the lexicon entries, compares
them with ALINE or PanPhon segmental distances, and blends in a dedicated tonal penalty.

## Features

- Convert Mandarin text to approximated IPA segments and tone numbers.
- Compute segmental distances with either PanPhon or ALINE.
- Add tone-specific penalties on top of segmental similarity.
- Replace ASR substrings with canonical surface forms from a lexicon when the combined
distance drops below a configurable threshold.

## Installation

The project follows standard Python packaging conventions and can be installed in editable
mode during development:

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

This installs the library as well as optional development dependencies such as `pytest`.

## Usage

```python
from asr_corrector import (
    ASRPostCorrector,
    CorrectionConfig,
    DistanceMetric,
    LexiconEntry,
)

lexicon = [
    LexiconEntry("李四"),
    LexiconEntry("石墨烯"),
]

config = CorrectionConfig(
    threshold=1.2,
    lambda_tone=0.8,
    distance_metric=DistanceMetric.PANPHON,
    segmental_weight=1.0,
    tone_mismatch_penalty=0.6,
    max_window_size=4,
)

corrector = ASRPostCorrector(lexicon, config=config)

asr_output = "今天历史材料到了。"
corrected, details = corrector.correct(asr_output)
print(corrected)  # -> 今天李四材料到了。
print(details)
```

Switch to ALINE by setting `distance_metric=DistanceMetric.ALINE`. Reduce `threshold` for a
more conservative replacement strategy, increase `lambda_tone` to force stronger tone
agreement, or adjust `segmental_weight` and `tone_mismatch_penalty` to customise the balance
between segmental and tonal cues.

Each `Correction` entry in `details` contains the original substring, replacement text,
character span, the computed distance, and the metric that triggered the substitution. This
information can be logged to audit replacements or to tune hyperparameters offline.

## How it works

1. Convert candidate substrings and lexicon entries to IPA using `pypinyin` with a custom
   mapping that preserves Mandarin-specific distinctions.
2. Evaluate a segmental distance using ALINE or PanPhon on de-toned IPA sequences.
3. Compute a tone distance separately and combine it with the segmental score through
   configurable weights.
4. Replace substrings whose combined distance is below the threshold while iterating through
   the ASR hypothesis from left to right.

## Caveats

- The IPA mapping is tailored for Mainland Mandarin and may require refinement for dialectal
  variants.
- Current substring search assumes candidate replacements roughly match the character length
  of their correct forms. Extending the lexicon with explicit alternative spellings is
  recommended when the ASR system frequently inserts or deletes characters.
- Tone handling uses simple numeric differences; integrating richer tone sandhi modelling can
  further improve precision.

## Contributing

Issues and pull requests are welcome. Please run the linting or test suite of your choice
before submitting contributions.
