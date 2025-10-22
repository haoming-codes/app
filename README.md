# Phonetic RAG Utilities

This repository implements a lightweight retrieval-augmented correction pipeline for multilingual (Chinese/English) ASR transcripts. It converts text to IPA with `phonemizer`, projects phones to articulatory feature vectors with `panphon`, and combines several distance signals (segment edits, articulatory DTW, Mandarin tones, and English stress) to propose replacements from a knowledge base of frequent entity/jargon mis-recognitions.

## Installation

```bash
pip install -e .
```

> **Note:** The phonetic distance calculator depends on a fork of `abydos`. The editable install automatically pulls `git+https://github.com/denizberkin/abydos.git` because it is listed in `pyproject.toml`.

If `espeak-ng` data files live outside the default search path, point `ESPEAK_DATA_PATH` to them before running any code:

```bash
export ESPEAK_DATA_PATH=/usr/lib/x86_64-linux-gnu/espeak-ng-data
```

## Algorithm overview

1. **Phoneme extraction** – Each window of ASR output and each knowledge-base entry is converted to IPA with `phonemizer` (espeak-ng backend). Chinese characters are transliterated to Pinyin (tone numbers retained) before phonemization. Uppercase acronyms are spelled out letter by letter (`AG.AL → A G A L`).
2. **Articulatory projection** – IPA segments are tokenised with `panphon` and mapped to binary/ternary articulatory feature vectors.
3. **Segment distance** – Multiple phonetic edit metrics (e.g. `PhoneticEditDistance`, `ALINE`) score string-level differences. The component scores are mixed according to `aggregation.segment_metric_weights`.
4. **Dynamic time warping** – Feature vectors are compared with DTW (local metric selectable via `aggregation.dtw_local_metric`). The normalised path cost becomes another segment-level signal and is combined with the edit distances via `aggregation.feature_weight`.
5. **Suprasegmentals** – Mandarin tone numbers (from transliteration) and English stress marks (from the IPA output) are aligned with simple Levenshtein penalties. Tone penalties are divided by the number of Chinese characters involved; stress penalties are divided by the number of English words.
6. **Aggregation** – The three normalised components are blended with convex coefficients: `lambda_segment`, `lambda_tone`, and `lambda_stress`. The resulting score lives in `[0, 1]` when the component weights are set conservatively.
7. **Sliding window search** – For each knowledge-base entry, a token window spanning a similar number of tokens is compared against the ASR output. Windows below `window_threshold` become correction candidates. The `max_candidates` lowest-scoring suggestions are returned.

## Usage

### IPA inspection

```python
from phonetic_rag.phonetics import PhoneticTranscriber

transcriber = PhoneticTranscriber()
print(transcriber.ipa('美国'))        # → 'meikuoɜ'
print(transcriber.ipa('Mini Map'))    # → 'mɪnˌiːmæp'
```

### Distance computation with explicit hyperparameters

```python
from phonetic_rag.config import DistanceAggregationConfig
from phonetic_rag.distance import PhoneticDistanceCalculator

aggregation = DistanceAggregationConfig(
    segment_metric_weights={'phonetic_edit_distance': 0.7, 'aline': 0.3},
    feature_weight=0.5,            # trade-off between edit metrics and DTW
    dtw_local_metric='cosine',     # alternatives: 'euclidean', 'manhattan'
    lambda_segment=0.6,
    lambda_tone=0.25,
    lambda_stress=0.15,
    tone_weight=1.0,
    stress_weight=0.5,
)
calculator = PhoneticDistanceCalculator(aggregation)

breakdown = calculator.distance('Mini Map', 'minivan')
print(breakdown.total)             # aggregate score
print(breakdown.segment_metrics)   # individual edit distances
print(breakdown.feature_component) # DTW contribution
```

### Correction pipeline

```python
from phonetic_rag.config import DistanceAggregationConfig, PhoneticPipelineConfig
from phonetic_rag.corrector import PhoneticCorrector

aggregation = DistanceAggregationConfig(
    segment_metric_weights={'phonetic_edit_distance': 0.5, 'aline': 0.5},
    feature_weight=0.3,
    lambda_segment=0.65,
    lambda_tone=0.2,
    lambda_stress=0.15,
)
pipeline_config = PhoneticPipelineConfig(
    aggregation=aggregation,
    window_threshold=0.35,  # reject windows with higher scores
    max_candidates=3,
)

corrector = PhoneticCorrector(
    knowledge_base=['美国', '成哥', 'Mini Map'],
    config=pipeline_config,
)

for match in corrector.correct('他提到了mango和minivan'):
    print(match.original, '→', match.replacement, match.distance.total)
```

The `PhoneticCorrector` exposes a `best_match` shortcut if only the top candidate is required.

## Tests

Run the automated checks with:

```bash
pytest
```

The suite covers IPA conversion, configurable distance aggregation, and the sliding-window corrector.
