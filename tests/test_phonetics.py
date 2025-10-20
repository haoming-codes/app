from asr_postcorrection.phonetics import text_to_ipa_segments


def test_text_to_ipa_segments_preserves_length():
    text = "张三"
    segments = text_to_ipa_segments(text)
    assert len(segments) == len(text)
    assert segments[0].startswith("ʈʂ")
    assert segments[0].endswith("˥")
