from rag_corrector import Entity, EntityCorrector


def test_basic_correction():
    entities = [
        Entity(surface="张三"),
        Entity(surface="北京", aliases=["背景"]),
    ]
    corrector = EntityCorrector(entities, tone_weight=0.5, window_slack=1, score_threshold=0.5)

    transcript = "今天章三去了背景旅游"
    corrected, corrections = corrector.correct(transcript)

    assert corrected == "今天张三去了北京旅游"
    assert len(corrections) == 2
    replaced_spans = {(c.original, c.replacement) for c in corrections}
    assert ("章三", "张三") in replaced_spans
    assert ("背景", "北京") in replaced_spans


def test_skip_exact_match():
    entities = [Entity(surface="李雷")]
    corrector = EntityCorrector(entities)

    transcript = "李雷韩梅梅都来了"
    corrected, corrections = corrector.correct(transcript)

    assert corrected == transcript
    assert corrections == []
