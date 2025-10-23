# bilingual-ipa

A small utility for turning Chinese/English bilingual text into its IPA representation using [`epitran`](https://github.com/dmort27/epitran).

## Installation

```bash
pip install -e .
```

## Usage

```python
from bilingual_ipa import text_to_ipa

ipa = text_to_ipa("Hello你好")
print(ipa)
```

`text_to_ipa` automatically detects Chinese and English segments, transliterates Chinese text with `Epitran('cmn-Hans', cedict_file='cedict_ts.u8', tones=True)` and English text with `Epitran('eng-Latn')`, and returns the combined IPA transcription.
