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

`text_to_ipa` automatically detects Chinese and English segments, calls `epitran.Epitran(...).transliterate` separately for each language with the correct language code (`cmn-Hans` for Chinese and `eng-Latn` for English), and returns the combined IPA transcription. Any additional keyword arguments are forwarded to `Epitran.transliterate`.
