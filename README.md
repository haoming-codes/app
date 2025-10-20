# rag-phonetic-correction

Tools for correcting Chinese ASR transcription errors by comparing ASR substrings with a curated lexicon of entities or jargon using phonetic and tonal distance. The package is designed to sit on top of an external ASR API, so it works with raw text transcripts without needing access to internal encoder or decoder states.

## Approach

1. **Syllable extraction** – Convert Chinese characters to Hanyu Pinyin with tone numbers using `pypinyin`. Strip tone numbers to obtain base syllables, and approximate each syllable with an IPA sequence built from handcrafted mappings of initials and finals.
2. **Segmental distance** – Run an edit distance over the toneless IPA syllable sequence. The default `PanphonFeatureDistance` uses PanPhon's feature edit distance, but wrappers are provided for Abydos' `PhoneticEditDistance` and `ALINE` metrics if those dependencies are available in your environment.
3. **Tone distance** – Compute a second edit distance over tone sequences using a confusion-weighted cost matrix. Tone penalties are normalised by the longer sequence length so they can be combined with segmental scores.
4. **Aggregation** – Combine segmental and tonal costs with tunable weights. A match is accepted when the combined score falls below a configurable threshold. Greedy post-processing keeps the lowest-cost non-overlapping matches so the corrected text can be reconstructed.

This mirrors the proposed RAG-style pipeline where each lexicon entry acts as a retrieval target and the edit distances rank possible replacements. The implementation separates IPA segment scores from tone confusion penalties, avoiding issues with tone diacritics inside IPA tokens.

## Installation

```bash
pip install -e .
```

The editable install keeps your working tree importable while you iterate on the distance weights or tone confusion matrix.

## Usage

```python
from rag_phonetic_correction import (
    LexiconEntry,
    PanphonFeatureDistance,
    PhoneticConverter,
    PhoneticMatcher,
    DistanceCalculator,
    ToneDistance,
)

# Hyperparameters
SEGMENT_WEIGHT = 1.0
TONE_WEIGHT = 0.3
THRESHOLD = 0.55
CONFUSION = {
    (1, 2): 0.4,
    (2, 3): 0.35,
    (3, 4): 0.5,
    (1, 4): 0.7,
}

converter = PhoneticConverter()
segment_metric = PanphonFeatureDistance()
tone_metric = ToneDistance(confusion_costs=CONFUSION, substitution_default=0.9)
calculator = DistanceCalculator(
    segmental=segment_metric,
    tone=tone_metric,
    segment_weight=SEGMENT_WEIGHT,
    tone_weight=TONE_WEIGHT,
)
matcher = PhoneticMatcher(
    converter=converter,
    distance_calculator=calculator,
    default_threshold=THRESHOLD,
)

lexicon = [
    LexiconEntry(term="微软"),
    LexiconEntry(term="阿里巴巴", threshold=0.4),  # entry-specific override
]

asr_output = "我们讨论了美软和阿狸巴巴的最新产品"
corrected, matches = matcher.correct_text(asr_output, lexicon)
print(corrected)  # -> 我们讨论了微软和阿里巴巴的最新产品
for match in matches:
    print(match.original, "→", match.replacement, "score=", match.distance)
```

### Switching distance metrics

```python
from rag_phonetic_correction import AbydosALINEDistance, AbydosPhoneticDistance

segment_metric = AbydosPhoneticDistance(qval=1.0)  # requires abydos with numpy>=2.0
# or
segment_metric = AbydosALINEDistance(scale=0.7)
```

If `abydos` fails to import (for example because of incompatible numpy builds), these wrappers raise an informative error while the PanPhon-based default keeps working.

### Custom tone penalties

Tone confusions vary by speaker and microphone. The `ToneDistance` class accepts your own confusion map, insertion/deletion costs, and substitution defaults so you can plug in confusion matrices derived from held-out ASR logs.

```python
ToneDistance(
    confusion_costs={(2, 3): 0.3, (3, 2): 0.3},
    substitution_default=1.2,
    insertion_cost=0.8,
    deletion_cost=0.8,
)
```

## Testing

```bash
pytest
```

The sample tests illustrate how to configure weights and thresholds so that near-miss transcriptions are corrected while dissimilar substrings are left untouched.
