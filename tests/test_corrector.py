from asr_corrector import (
    ASRCorrector,
    CorrectionConfig,
    KnowledgeBase,
    KnowledgeEntry,
    MetricConfig,
    ToneConfig,
)


def test_corrector_applies_expected_replacements():
    knowledge_base = KnowledgeBase([
        KnowledgeEntry("AG.AL"),
        KnowledgeEntry("清华大学"),
    ])
    config = CorrectionConfig(
        metrics=[MetricConfig(name="panphon", weight=1.0)],
        tone=ToneConfig(weight=1.0, default_cost=1.0),
        threshold=1.5,
    )
    corrector = ASRCorrector(knowledge_base, config)
    text = "我们访问了清华大學 并跟 AGAL 团队交流"
    result = corrector.correct(text)
    assert result.text == "我们访问了清华大学 并跟 AG.AL 团队交流"
    assert {r.replacement for r in result.replacements} == {"清华大学", "AG.AL"}
    assert all(r.breakdown.total <= config.threshold for r in result.replacements)


def test_distance_calculator_exposed_for_tuning():
    knowledge_base = KnowledgeBase([KnowledgeEntry("量子纠缠")])
    config = CorrectionConfig(metrics=[MetricConfig("panphon", weight=1.0)])
    corrector = ASRCorrector(knowledge_base, config)
    breakdown = corrector.distance_calculator.distance("量子纠缠", "量子纠纏")
    assert breakdown.total == breakdown.segmental
    assert breakdown.total >= 0.0
