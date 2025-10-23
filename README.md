# bilingual-ipa

A small utility for turning Chinese/English bilingual text into its IPA representation using [`eng_to_ipa`](https://github.com/mphilli/English-to-IPA) for English and [`dragonmapper`](https://github.com/tsroten/dragonmapper) for Chinese.

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

`text_to_ipa` automatically detects Chinese and English segments, converts English text with `eng_to_ipa.convert`, converts Chinese Hanzi with `dragonmapper.hanzi.to_ipa`, and returns the combined IPA transcription.
