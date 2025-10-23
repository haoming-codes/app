# bilingual-ipa

A small utility for turning Chinese/English bilingual text into its IPA representation using [`eng_to_ipa`](https://github.com/mphilli/English-to-IPA) for English and [`dragonmapper`](https://github.com/tsroten/dragonmapper) for Chinese.

## Installation

```bash
pip install -e .
```

## Usage

```python
from bilingual_ipa import text_to_ipa

ipa_result = text_to_ipa("Hello你好")
print(ipa_result.phones)
print(ipa_result.tone_marks)
print(ipa_result.stress_marks)
print(ipa_result.syllable_counts)
```

`text_to_ipa` automatically detects Chinese and English segments, converts English text with `eng_to_ipa.convert`, converts Chinese Hanzi with `dragonmapper.hanzi.to_ipa`, and returns an `IPAConversionResult` dataclass containing phones, tone marks, stress marks, and syllable counts for each English word or Chinese character in the text.
