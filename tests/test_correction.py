import math

from rag_asr_correction import (
    CorrectionConfig,
    CorrectionEngine,
    KnowledgeBase,
    SegmentalMetric,
    SegmentalMetricConfig,
    ToneDistanceConfig,
)
from rag_asr_correction.distance import DistanceComputer
from rag_asr_correction.phonetics import build_phonetic_representation


def make_config() -> CorrectionConfig:
    return CorrectionConfig(
        segmental_metrics=[
            SegmentalMetricConfig(SegmentalMetric.PANPHON_WEIGHTED, weight=1.0),
            SegmentalMetricConfig(SegmentalMetric.ABYDOS_PHONETIC_EDIT, weight=0.4),
            SegmentalMetricConfig(SegmentalMetric.ABYDOS_ALINE, weight=0.6),
        ],
        tone_config=ToneDistanceConfig(
            weight=1.2,
            substitution_costs={
                (1, 2): 0.4,
                (2, 3): 0.6,
                (3, 4): 0.6,
                (4, 5): 0.8,
            },
            insertion_cost=0.5,
            deletion_cost=0.5,
        ),
        threshold=5.0,
        tradeoff_lambda=0.8,
        max_window_size=4,
    )


def test_distance_comparison():
    config = make_config()
    computer = DistanceComputer(config)

    same = computer.distance_between_strings("AlphaFold", "AlphaFold", lang_a="en-us", lang_b="en-us")
    similar = computer.distance_between_strings("AlphaFold", "Alpha Field")
    chinese_same = computer.distance_between_strings("张伟", "张伟", lang_a="cmn", lang_b="cmn")
    chinese_diff = computer.distance_between_strings("张微", "张伟", lang_a="cmn", lang_b="cmn")

    assert math.isclose(same, 0.0, abs_tol=1e-6)
    assert chinese_same <= 1e-6
    assert similar > same
    assert chinese_diff > chinese_same


def test_tone_distance_penalizes_differences():
    config = make_config()
    computer = DistanceComputer(config)

    rep_a = build_phonetic_representation("妈妈", language="cmn")
    rep_b = build_phonetic_representation("麻麻", language="cmn")
    rep_c = build_phonetic_representation("妈妈", language="cmn")

    tone_diff = computer.tone_distance(rep_a, rep_b)
    tone_same = computer.tone_distance(rep_a, rep_c)

    assert tone_same == 0.0
    assert tone_diff > 0.0


def test_correction_engine_replaces_best_matches():
    config = make_config()
    kb = KnowledgeBase(["张伟", "AG.AL"])
    engine = CorrectionEngine(kb, config)

    text = "今天我们采访了张微博士和AGAL团队。"
    result = engine.correct(text)

    assert result.corrected == "今天我们采访了张伟博士和AG.AL团队。"
    assert len(result.applied) == 2
    assert {candidate.entry.text for candidate in result.applied} == {"张伟", "AG.AL"}
