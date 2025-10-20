# Phonetic RAG utilities

This repository provides tools for post-processing multilingual (Mandarin / English) ASR
transcripts using a retrieval-augmented correction pipeline. The focus is on
recovering named entities and jargon that are commonly mis-transcribed by the
recogniser. The core idea is to compute a combined phonetic + tonal distance
between substrings in the ASR hypothesis and a lexicon of reference forms, then
suggest replacements whose distance falls below a configurable threshold.

The implementation follows these design choices:

* Mandarin syllables are converted to IPA using a rule-based Pinyin to IPA map
  and tones are extracted separately for use in a tone-aware edit distance.
* English tokens are converted to IPA with `eng_to_ipa`; they contribute only to
  the segmental distance.
* Segmental distances can be computed using PanPhon weighted feature edit
  distance, Abydos' `PhoneticEditDistance`, `ALINE`, or a CLTS feature vector Euclidean distance.
* Tone mismatches are scored independently with a confusion-weighted edit
  distance so that tone errors can be penalised differently from segmental
  deviations.

## Installation

Create a virtual environment and install in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

> **Note:** The project depends on a fork of Abydos that contains recent fixes.
> The dependency is specified as
> `abydos @ git+https://github.com/denizberkin/abydos.git` in `pyproject.toml` so
> `pip` will download it automatically.

## Usage

### Computing distances directly

Use the convenience helper to explore the effect of different hyperparameters:

```python
from phonetic_rag import DistanceConfig, ToneConfusionMatrix, ToneDistanceConfig, compute_distance

config = DistanceConfig(
    segment_metric="panphon",  # panphon | phonetic_edit | aline | clts
    tradeoff_lambda=0.65,       # weight for segmental distance
    threshold=1.2,
    tone_config=ToneDistanceConfig(
        strategy="weighted",
        confusion=ToneConfusionMatrix(
            substitution_costs={(1, 2): 0.35, (2, 3): 0.45},
            insertion_cost=0.9,
            deletion_cost=0.9,
        ),
        normalize=True,
    ),
)

score = compute_distance("美國", "美国", config=config)
print(score)
```

Switching the segment metric or tone strategy is as simple as changing
`segment_metric` or `tone_config.strategy` to `"none"`.

### Matching a transcript against a glossary

```python
from phonetic_rag import DistanceConfig, PhoneticMatcher

lexicon = ["莫德納", "輝瑞", "OpenAI", "量子計算"]
config = DistanceConfig(segment_metric="aline", tradeoff_lambda=0.8, threshold=1.1)
matcher = PhoneticMatcher(dictionary=lexicon, config=config)

transcript = "我們今天討論莫得納疫苗的最新數據"
for match in matcher.match(transcript):
    print(match)
```

`PhoneticMatcher.match` returns `MatchResult` objects sorted by ascending
distance. You can tune:

* `DistanceConfig.tradeoff_lambda` &ndash; balances segmental (λ) vs tonal (1−λ)
  contributions.
* `DistanceConfig.threshold` &ndash; maximum acceptable combined distance for a
  match.
* `DistanceConfig.segment_metric` &ndash; selects `panphon`, `phonetic_edit`,
  `aline`, or `clts`.
* `ToneDistanceConfig` parameters &ndash; confusion weights, insertion/deletion
  costs, and normalisation behaviour.

### RAG-style correction loop

1. Build a lexicon of domain entities and jargon.
2. Run `PhoneticMatcher.match` on each ASR hypothesis.
3. For candidates below the configured threshold, surface the dictionary entry
   as a correction suggestion.
4. Optionally verify suggestions using additional context or a language model
   before rewriting the transcript.

The modular configuration makes it straightforward to experiment with different
phonetic distance definitions and tone penalties during hyperparameter search.
