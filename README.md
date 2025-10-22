# bilingual-ipa

A small utility for turning Chinese/English bilingual text into its IPA representation using [`phonemizer`](https://github.com/bootphon/phonemizer) with the `espeak` backend.

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

`text_to_ipa` automatically detects Chinese and English segments, calls `phonemizer.phonemize` separately for each language with the correct language code (`cmn` for Chinese and `en` for English), and returns the combined IPA transcription. You can pass any additional keyword arguments supported by `phonemizer.phonemize` (such as `strip=True`) and they will be forwarded automatically.
