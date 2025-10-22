from asr_correction.phonetics import IPAConverter


def test_sequence_contains_tone_and_stress():
    converter = IPAConverter()
    seq = converter.sequence('美国 Mini Map')
    assert seq.ipa.startswith('meɪ')
    assert seq.tones()[:2] == [3, 2]
    assert 'primary' in seq.stresses()


def test_chinese_sequence_preserves_length():
    converter = IPAConverter()
    seq = converter.sequence('成哥')
    # Ensure we produced two syllables and tone annotations
    assert len(seq.tones()) == 2
    assert seq.tones() == [2, 1]
    assert len(seq.phones) >= 4
