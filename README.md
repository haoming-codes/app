# Phonetic Correction Toolkit

Utilities for correcting multilingual (Chinese/English) automatic speech recognition (ASR)
transcripts by comparing them against a knowledge base of entities and jargon. The library
implements the phonetic-distance pipeline described in the project brief and exposes helpers
for IPA inspection, configurable distance scoring, and sliding-window matching against a
vocabulary of trusted spellings.

## Installation

```bash
pip install -e .[test]
```

The package depends on [`espeak-ng`](https://github.com/espeak-ng/espeak-ng) via
[`phonemizer`](https://github.com/bootphon/phonemizer) for multilingual phonemization.
The container used for evaluation already ships with the binary; if you are running this
locally make sure that `espeak-ng` is available on the system path.

## Algorithm overview

1. **Phonemization**
   - Chinese characters are phonemized with the `espeak-ng` backend using the
     `cmn-latn-pinyin` voice. Pinyin digits are stripped from the IPA stream but retained
     as tone annotations via `pypinyin`.
   - English words are phonemized with the American English voice (`en-us`) with stress
     markers enabled. All-cap tokens are interpreted as acronyms by spelling out their
     letters prior to phonemization.
   - Tokens that are not Chinese characters or Latin words are ignored during distance
     scoring.
2. **Feature representation**
   - Each output string is segmented into phones using `panphon`. Every phone is mapped to
     a 24-dimensional articulatory feature vector (with `+/-/0` encoded as `1/-1/0`).
3. **Segment distances**
   - Multiple segment-level metrics can be combined by weight: phonetic edit distance,
     ALINE, and dynamic time warping (DTW) over the articulatory feature vectors. DTW
     costs are averaged along the optimal path and squashed to the `[0, 1]` interval.
4. **Tone and stress penalties**
   - Mandarin tone sequences are obtained per Han character. An edit distance between the
     tone sequences contributes to the final score, scaled by `tone_penalty` and normalized
     by the number of Chinese characters involved.
   - English stress is tracked per word: primary stress (`ˈ`) contributes a level of `1`,
     secondary stress (`ˌ`) contributes `2`, and unstressed words contribute `0`. The edit
     distance between stress sequences is scaled by `stress_penalty` and normalized by the
     number of English words.
5. **Combination**
   - The normalized segment, tone, and stress components are combined with a convex blend
     controlled by `lambda_segment`, `lambda_tone`, and `lambda_stress`.
6. **Windowed knowledge-base search**
   - ASR outputs are tokenized into Chinese characters and Latin word spans. The matcher
     slides windows across the transcript, rescores each window against every knowledge
     base entry, and keeps the best match per entry if it falls below the configured
     threshold.

## Usage

```python
from phonetic_correction import DistanceConfig, PhoneticMatcher

config = DistanceConfig(
    segment_metrics={
        "phonetic_edit_distance": 0.4,
        "aline": 0.3,
        "dtw": 0.3,
    },
    lambda_segment=0.55,
    lambda_tone=0.3,
    lambda_stress=0.15,
    tone_penalty=1.2,
    stress_penalty=0.8,
    threshold=0.5,
    window_expansion=2,
)
matcher = PhoneticMatcher(config)
```

### Inspecting IPA transcriptions

```python
transcription = matcher.transcribe("美国 Mini Map")
print(transcription.ipa)            # -> meikuoɜmɪnimæp
print(transcription.tone_sequence)  # -> [3, 2]
print(transcription.stress_sequence)  # -> [1, 1]
```

### Computing distances between snippets

```python
distance = matcher.distance("mango", "美国")
print(distance.segment_distance)
print(distance.tone_distance)
print(distance.stress_distance)
print(distance.total_distance)
```

You can also access the lower-level calculator directly for hyperparameter sweeps:

```python
from phonetic_correction import PhoneticDistanceCalculator
phonemizer = matcher.phonemizer
calculator = PhoneticDistanceCalculator(config)
first = phonemizer.transcribe("成哥")
second = phonemizer.transcribe("恒哥")
print(calculator.distance(first, second))
```

### Scanning an ASR transcript against a knowledge base

```python
knowledge_base = [
    "美国",
    "成哥",
    "Mini Map",
]
transcript = "我们讨论了 mango 然后打开了 minivan 查看地图"
for match in matcher.match(transcript, knowledge_base):
    print(match.candidate, "<->", match.window_text, match.distance.total_distance)
```

Adjust `threshold`, `window_expansion`, and the metric weights to tune precision/recall for
your specific ASR model and knowledge base. The `PhoneticTranscription` objects returned by
`transcribe` expose IPA strings, tone sequences, stress sequences, and per-phone feature
vectors to make hyperparameter tuning and debugging straightforward.
