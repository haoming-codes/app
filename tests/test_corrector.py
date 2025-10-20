from asr_correction.config import DistanceComputationConfig, SegmentalMetricConfig, ToneDistanceConfig
from asr_correction.corrector import Corrector


BASIC_CONFIG = DistanceComputationConfig(
    segmental_metrics=[SegmentalMetricConfig("phonetic_edit_distance")],
    tone=ToneDistanceConfig(weight=1.0, confusion_penalties={("2", "3"): 0.2}),
    tradeoff_lambda=0.7,
    threshold=0.55,
)


def test_suggests_knowledge_base_match():
    kb = ["阿里巴巴", "DeepMind", "AG.AL"]
    corrector = Corrector(kb, config=BASIC_CONFIG)
    output = "今天我们讨论了阿里爸爸的最新成果"
    suggestions = corrector.suggest(output)
    assert any(s.replacement == "阿里巴巴" for s in suggestions)
    best = suggestions[0]
    assert best.replacement in kb
    assert best.distance <= BASIC_CONFIG.threshold


def test_acronym_is_expanded_for_distance():
    kb = ["AG.AL"]
    corrector = Corrector(kb, config=BASIC_CONFIG)
    output = "今天提到的A G A L很重要"
    suggestions = corrector.suggest(output)
    assert suggestions
    assert suggestions[0].replacement == "AG.AL"


def test_apply_best_replaces_text():
    kb = ["阿里巴巴"]
    corrector = Corrector(kb, config=BASIC_CONFIG)
    output = "我们和阿里爸爸合作"
    result = corrector.apply_best(output)
    assert "阿里巴巴" in result
