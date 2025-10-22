import pytest

from phonetic_rag.phonetics import PhoneticTranscriber


def test_transcriber_handles_mixed_text():
    transcriber = PhoneticTranscriber()
    rep = transcriber.transcribe('美国 Mini Map')
    assert rep.ipa
    assert rep.tone_sequence == [3, 2]
    assert rep.chinese_char_count == 2
    assert rep.english_word_count >= 1


def test_acronym_expansion_changes_ipa():
    transcriber = PhoneticTranscriber()
    ipa_acronym = transcriber.ipa('AG', treat_all_caps_as_acronyms=True)
    ipa_word = transcriber.ipa('AG', treat_all_caps_as_acronyms=False)
    assert ipa_acronym != ipa_word
    assert 'dʒ' in ipa_acronym
