import re

from ragasr.phonetics import ipa_transcription, tokenize_text, tones_for


def test_tokenize_mixed_text():
    tokens = tokenize_text("成哥 去 Mango")
    kinds = [t.kind for t in tokens]
    assert kinds == ["cjk", "cjk", "space", "cjk", "space", "latin"]


def test_ipa_transcription_non_empty():
    ipa = ipa_transcription("美国 Mini Map")
    assert isinstance(ipa, str)
    assert ipa
    assert re.search(r"[a-zɡɯɪ]", ipa)


def test_tones_for_chinese_characters():
    tones = tones_for("成哥")
    assert len(tones) == 2
    assert all(isinstance(t, int) for t in tones)
