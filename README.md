# ASR Postprocessor

This repository implements a lightweight, RAG-inspired post-processing layer
for Chinese automatic speech recognition (ASR) output. The focus is on
repairing named-entity recognition mistakes by matching the raw transcription
against a curated lexicon using phonetic similarity (IPA + panphon feature
space) instead of relying on orthographic similarity. This design keeps the
post-processor model-agnostic, which is appropriate when the ASR system is only
accessible through an API.

## Why phonetic matching?

1. **Model-agnostic correction** – We treat the ASR output as plain text and
   avoid any dependence on model internals (encoder states, logits, etc.).
2. **Robustness to common entity errors** – Chinese named entities often share
   similar syllable shapes. Matching in IPA space allows us to recover intended
   entities even when the ASR system produces homophones or near-homophones.
3. **Tone-aware scoring** – Leveraging `panphon` provides a fine-grained feature
   representation that captures tone differences and articulatory similarity
   (e.g. /s/ vs. /z/ being closer than /s/ vs. /k/), addressing a limitation of
   pinyin string matching.

The correction engine slides a phonetic window across the ASR hypothesis,
computes panphon-based similarities against each lexicon entry, and replaces
high-confidence matches with their canonical surfaces. Hyper-parameters allow
fine-grained control over the acceptance threshold and the balance between pure
phonetic similarity and structural heuristics such as length consistency.

## Installation

The project follows a standard `pyproject.toml` layout and can be installed in
editable mode:

```bash
pip install --upgrade pip
pip install -e .
```

This installs the package under the name `asr-postprocessor`.

## Usage

```python
from asr_postprocessor import (
    ChinesePhoneticizer,
    CorrectionConfig,
    CorrectionEngine,
    NamedEntity,
    PhoneticLexicon,
)

# Build a lexicon of entities you care about. IPA values are optional – they
# will be generated automatically when omitted.
lexicon = PhoneticLexicon(ChinesePhoneticizer())
lexicon.extend(
    [
        NamedEntity(surface="清华大学"),
        NamedEntity(surface="张伟"),
        NamedEntity(surface="阿里巴巴"),
    ]
)

# Configure the correction engine with explicit hyper-parameters.
config = CorrectionConfig(
    threshold=0.78,        # minimum overall score required to accept a match
    tradeoff_lambda=0.9,   # weight on phonetic similarity vs. length heuristic
    length_slack=1,        # allow +/- one syllable when forming candidates
    length_penalty=0.3,    # penalty strength for length deviations
)
engine = CorrectionEngine(lexicon=lexicon, config=config)

asr_output = "他毕业于清华大學后来加入阿里爸爸"
corrected, matches = engine.correct(asr_output)
print(corrected)
for match in matches:
    print(match.matched_text, "->", match.entity.surface, match.score)
```

## Extending the approach

- Provide richer metadata per entity (e.g. type, priority) through the
  `NamedEntity.metadata` field and adjust replacement logic accordingly.
- Add on-disk persistence for the lexicon to share the curated entity list.
- Integrate additional heuristics (language model rescoring, contextual filters)
  in the candidate selection phase for even higher precision.

## License

This project is distributed under the MIT License.
