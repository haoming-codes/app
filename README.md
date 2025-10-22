# Bilingual IPA Conversion

This package provides a simple helper for phonemizing bilingual Chinese/English
text into IPA symbols. Text is split into language-specific segments and fed to
[`phonemizer`](https://github.com/bootphon/phonemizer) using the `espeak-ng`
backend with the appropriate language codes (`cmn` for Mandarin Chinese and
`en` for English).

## Installation

Install the package in editable mode with:

```bash
pip install -e .
```

This will also install the `phonemizer` dependency.

## Usage

```python
from bilingual_ipa import text_to_ipa

ipa = text_to_ipa("你好 world", separator=" ")
print(ipa)  # -> IPA strings for the Chinese and English portions
```

All keyword arguments are passed through to
`phonemizer.phonemize`, allowing full control over its behaviour.
Non-language characters such as punctuation are preserved as-is in the output.

## Testing

Run the test suite with:

```bash
pytest
```
