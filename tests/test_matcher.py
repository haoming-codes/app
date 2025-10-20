from asr_postcorrection import CorrectionEngine, LexiconEntry


LEXICON = [
    LexiconEntry("王菲"),
    LexiconEntry("张学友"),
]


def test_corrects_named_entity() -> None:
    engine = CorrectionEngine(LEXICON, threshold=0.5)
    result = engine.correct("我喜欢王非的歌")
    assert result == "我喜欢王菲的歌"


def test_skips_low_similarity() -> None:
    engine = CorrectionEngine(LEXICON, threshold=0.9)
    text = "今天去了王府井"
    assert engine.correct(text) == text


def test_update_lexicon() -> None:
    engine = CorrectionEngine([LexiconEntry("李雷")])
    engine.update_lexicon([LexiconEntry("韩梅梅")])
    result = engine.correct("李镭和韩妹妹来了")
    assert "李雷" in result
    assert "韩梅梅" in result
