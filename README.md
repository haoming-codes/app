# RAG-ASR Corrector

Tools for correcting named entities and jargon in multilingual (Chinese/English) ASR transcripts using phonetic distance search over a knowledge base.

## Algorithm overview

1. **Knowledge base** – Maintain a list of entity strings (Chinese characters and English words/jargon). All-uppercase entries are treated as spelled-out acronyms.
2. **Phonetic projection** – Convert both ASR substrings and knowledge entries to a common IPA space using [`phonemizer`](https://github.com/bootphon/phonemizer) with the `espeak` backend:
   - Chinese characters are first romanised with tone digits via `pypinyin` and then rendered with the `cmn-latn-pinyin` voice.
   - English words are rendered with `en-us` pronunciations while preserving primary/secondary stress marks.
   - The IPA string is segmented and converted into articulatory feature vectors using `panphon`.
3. **Distance computation** – For each candidate window in the ASR text and each knowledge-base entry:
   - Segment-level similarities with [`abydos`](https://github.com/denizberkin/abydos) (`PhoneticEditDistance`, `ALINE`).
   - Feature-DTW alignment on articulatory feature vectors using either cosine or Euclidean local costs.
   - Tone penalties (Mandarin syllable tones) via edit distance normalised by the number of Chinese characters.
   - Stress penalties (English word stress levels) via edit distance normalised by the number of English words.
   - Combine the normalised components with user-provided weights.
4. **Suggestion search** – Slide windows across the ASR transcript (at the granularity of Chinese characters and English words) and compute the combined distance to each knowledge entry. Results below the configured threshold are returned as correction suggestions.

## Installation

The project ships as a standard Python package and supports editable installs:

```bash
pip install -e .
```

> **Note**: Ensure `espeak-ng` is available on the system so that `phonemizer` can render IPA for Mandarin and English. On Debian/Ubuntu systems: `apt-get install espeak-ng`.

To run the test suite:

```bash
pip install -e .[test]
pytest
```

## Usage

```python
from rag_asr_corrector import (
    CorrectionConfig,
    DistanceConfig,
    KnowledgeBase,
    PhoneticCorrector,
)

kb = KnowledgeBase()
kb.extend([
    "美国",  # The United States
    "成哥",
    "Mini Map",
])

config = CorrectionConfig(
    distance=DistanceConfig(
        segment_metrics=["phonetic_edit_distance", "aline"],
        feature_metric="cosine",
        segment_weight=0.55,
        feature_weight=0.3,
        tone_weight=0.1,
        stress_weight=0.05,
    ),
    threshold=0.22,        # reject candidates above this combined score
    window_tolerance=1,    # allow +/- 1 token window around entity length
)

corrector = PhoneticCorrector(kb, config=config)
text = "他们去了mango并打开了minivan"
corrected_text, applied = corrector.apply(text)
for suggestion in applied:
    print(suggestion.replacement, suggestion.breakdown.total)
```

### Distance-only utilities

You can compute phonetic distances or inspect IPA projections directly to tune hyperparameters:

```python
from rag_asr_corrector import DistanceCalculator, DistanceConfig, MultilingualIPAConverter

converter = MultilingualIPAConverter()
print(converter.ipa_string("Mini Map"))  # IPA string without spaces

calc = DistanceCalculator(DistanceConfig(stress_weight=0.1))
breakdown = calc.distance("美国", "mango")
print(breakdown.segment_distance, breakdown.feature_distance, breakdown.tone_distance, breakdown.total)
```

### Customising the search

* Increase `window_tolerance` to consider larger context windows around each entity.
* Adjust the component weights inside `DistanceConfig` to favour segment similarity, articulatory alignment, or suprasegmental cues.
* Tighten or relax the `threshold` depending on the desired precision/recall trade-off.

## Project structure

- `rag_asr_corrector/ipa.py` – multilingual IPA conversion and feature extraction.
- `rag_asr_corrector/distances.py` – phonetic distance aggregation, including DTW and tone/stress penalties.
- `rag_asr_corrector/corrector.py` – sliding-window matcher over transcripts producing correction suggestions.
- `rag_asr_corrector/tests/` – pytest suite covering IPA conversion, distance metrics, and correction suggestions.

## License

MIT (see `LICENSE` if provided by the downstream user).
