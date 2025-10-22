# ragasr

A toolkit for correcting multilingual ASR transcription errors with phonetic distance
matching against a knowledge base of named entities and jargon. The implementation
uses phonetic and articulatory feature distances to propose corrections.

## Installation

Install in editable mode to experiment with configuration and hyperparameters:

```bash
pip install -e .
```

> **Note**: The project depends on a fork of `abydos`. Installing the package will
> automatically fetch `git+https://github.com/denizberkin/abydos.git`.

## Algorithm overview

Given an ASR hypothesis and a knowledge base entry, the correction algorithm performs
four stages:

1. **Tokenization and language hints** – mixed Chinese/English text is split into
   content tokens (Chinese characters, English words, acronyms, digits) and
   separators (spaces and punctuation). All-capital tokens are treated as spelled-out
   acronyms.
2. **Phonetic transcription** – each content token is phonemized with `espeak-ng`
   via `phonemizer`: Chinese characters with the `cmn` backend and English tokens
   with the `en-us` backend. Acronyms are expanded to letter sequences before
   phonemization. The resulting IPA strings drop whitespace and language tags.
3. **Feature extraction and alignment** – IPA strings are segmented with `panphon`
   and converted to articulatory feature vectors. Mandarin tones are extracted via
   `pypinyin`, while primary/secondary English stress marks come directly from the
   IPA transcription. Segment sequences are aligned with a combination of
   `PhoneticEditDistance`, `ALINE`, and dynamic time warping (DTW) on the feature
   vectors.
4. **Distance aggregation** – segment, tone, and stress distances are normalised by
   their alignment length and combined with configurable weights. A light heuristic
   penalises large window-length mismatches and windows that mix Chinese and English
   content when the entry itself is monolingual. When the combined score falls below
   a threshold, the window is replaced with the knowledge base entry (or its canonical
   form). Conflicts are resolved greedily from left to right.

## Usage

```python
from ragasr import (
    ASRCorrector,
    KnowledgeEntry,
    CorrectionConfig,
    DistanceConfig,
    PhoneticDistanceCalculator,
    ipa_transcription,
)

kb = [
    KnowledgeEntry("美国"),
    KnowledgeEntry("成哥"),
    KnowledgeEntry("Mini Map"),
]

# Fine-tune hyperparameters to match your use case.
distance_config = DistanceConfig(
    threshold=0.4,
    segment_metrics=("phonetic", "aline"),
    feature_metric="cosine",
    use_feature_dtw=True,
    segment_weight=0.55,
    tone_weight=0.25,
    stress_weight=0.20,
    tone_penalty=0.8,
    stress_penalty=1.2,
)

config = CorrectionConfig(distance=distance_config, window_radius=2)
corrector = ASRCorrector(kb, config=config)

asr_output = "成哥去 Mango 看 mini van"
corrected_text, candidates = corrector.correct(asr_output)
print(corrected_text)
for candidate in candidates:
    print(candidate.entry.surface, candidate.distance)

# Direct access to the phonetic representations helps with tuning thresholds.
calculator = PhoneticDistanceCalculator(distance_config)
result = calculator.distance("mango", "美国")
print("Combined distance:", result.combined)
print("Segment component:", result.segment)
print("Tone component:", result.tone)
print("Stress component:", result.stress)

# Inspect IPA for a multilingual string.
print(ipa_transcription("成哥去 Mini Map"))
```

### Hyperparameter notes

- `DistanceConfig.threshold` controls how aggressive the replacements are. Lower
  values demand closer matches.
- `segment_metrics` chooses which symbolic distances are included; the default uses
  both `PhoneticEditDistance` and `ALINE`.
- `feature_metric` selects the local cost for DTW over articulatory features. Choose
  among `"cosine"`, `"manhattan"`, or `"euclidean"`.
- `segment_weight`, `tone_weight`, and `stress_weight` are λ parameters for the
  segment, tone, and stress components respectively. They should sum to approximately
  1.0 for an interpretable combined score.
- `tone_penalty` and `stress_penalty` scale the suprasegmental penalties when tone
  or stress mismatches are especially detrimental.
- `window_radius` controls how many neighbouring tokens are considered when sliding
  over the ASR output.
- Set `use_feature_dtw=False` if you want to exclude the articulatory DTW component
  and rely solely on `abydos` distances.

## Testing

Run the test suite after installing the project in editable mode:

```bash
pytest
```

## License

MIT
