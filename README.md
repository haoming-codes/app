# RAG-style phonetic post-correction for Chinese ASR

This repository implements a retrieval-and-generation inspired workflow for
cleaning up Chinese ASR transcripts. The focus is on recovering jargon and
named entities that are frequently mistranscribed. The system works purely
from API outputs (plain text hypotheses) and does **not** require access to
internal acoustic or embedding representations.

The pipeline follows these steps:

1. Convert both the transcript and a user-maintained lexicon of canonical
   forms into Mandarin IPA with tone annotations. Initial/final splitting is
   handled with `pypinyin` and a hand-crafted mapping to IPA.
2. Compare transcript substrings against lexicon entries with multiple
   segmental distance metrics (`panphon` feature edit distance, ALINE, and the
   Abydos phonetic edit distance) and a separate tone-distance component.
3. Combine segmental and tonal distances with user-configurable coefficients
   to produce a scalar score. Candidates below a threshold trigger a
   replacement in the transcript.

This matches the proposed approach while making two practical adjustments:

- Tones are aligned with a dynamic-programming edit distance so that partial
  mismatches receive fractional penalties instead of all-or-nothing scores.
- Zero-initial syllables (written with `y`/`w` in Hanyu Pinyin) are mapped to
  their underlying finals so that the segmental metrics operate on the correct
  IPA sequences.

## Installation

Install the project in editable mode so that changes to the source take effect
immediately:

```bash
pip install -e .
```

This pulls in the required dependencies (`pypinyin`, `panphon`, and
`abydos`).

## Usage

```python
from rag_phonetic_correction import (
    CorrectorConfig,
    LexiconEntry,
    PhoneticDistanceConfig,
    PhoneticRAGCorrector,
)

lexicon = [
    LexiconEntry(term="阿里巴巴"),
    LexiconEntry(term="华为"),
    LexiconEntry(term="Transformer", pronunciation=["chuan1", "si1", "mo4", "na4"]),
]

# Hyperparameters controlling the scoring function.
distance_cfg = PhoneticDistanceConfig(
    panphon_weight=1.0,
    aline_weight=0.3,
    phonetic_edit_weight=0.2,
    tone_weight=0.4,
    tone_gap_penalty=0.5,
    tone_default_penalty=0.8,
    normalization_exponent=1.2,
)

corrector_cfg = CorrectorConfig(
    distance=distance_cfg,
    threshold=1.1,
    max_matches_per_entry=2,
    allow_overlaps=False,
)

corrector = PhoneticRAGCorrector(lexicon, config=corrector_cfg)

transcript = "我们参观了阿里爸爸在杭州的总部, 还拜访了花为研究院。"
corrected, matches = corrector.correct(transcript)

print(corrected)
for match in matches:
    print(match.entry.entry.term, match.distance, transcript[match.start:match.end])
```

### Choosing thresholds and weights

- `threshold` controls how aggressive the correction is. Lower values only
  accept very close matches.
- `panphon_weight`, `aline_weight`, and `phonetic_edit_weight` adjust the
  trade-off between the three segmental metrics. Set unwanted metrics to `0`.
- `tone_weight`, `tone_gap_penalty`, and `tone_default_penalty` tune how tone
  mismatches influence the total distance.
- `normalization_exponent` changes the tone-distance normalization; values
  above 1.0 penalize longer sequences slightly less.

Because the system uses weighted sums, you can calibrate these parameters
against held-out transcripts and adjust the threshold accordingly.

## Development

After installing in editable mode you can run any local experiments directly.
The package exposes type information via `py.typed`, making it friendly to
static tooling such as `mypy` or `pyright`.
