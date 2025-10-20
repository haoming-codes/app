from asr_corrector import (
    Corrector,
    DistanceConfig,
    KnowledgeBase,
    KnowledgeEntry,
    SegmentalConfig,
    SegmentalMetric,
    ToneConfig,
    ToneMetric,
)
from asr_corrector.config import DEFAULT_TONE_CONFUSION


def test_corrector_suggests_best_match():
    config = DistanceConfig(
        segmental=SegmentalConfig(metric=SegmentalMetric.ABYDOS_PHONETIC),
        tone=ToneConfig(metric=ToneMetric.CONFUSION, confusion_costs=DEFAULT_TONE_CONFUSION),
        tone_weight=0.5,
        threshold=4.0,
    )
    kb = KnowledgeBase(
        [
            KnowledgeEntry("量子退火", "cmn-Hans"),
            KnowledgeEntry("quantum annealing", "eng-Latn"),
        ]
    )
    corrector = Corrector(kb, config)
    corrections = corrector.suggest("我们讨论的是quantum eniling技术")
    assert corrections
    best = corrections[0]
    assert best.replacement == "quantum annealing"
    new_text = corrector.apply_best("我们讨论的是quantum eniling技术")
    assert "quantum annealing" in new_text


def test_corrector_respects_threshold():
    config = DistanceConfig(
        segmental=SegmentalConfig(metric=SegmentalMetric.ABYDOS_PHONETIC),
        tone=ToneConfig(metric=ToneMetric.NONE),
        threshold=0.1,
    )
    kb = KnowledgeBase([KnowledgeEntry("阿里巴巴", "cmn-Hans")])
    corrector = Corrector(kb, config)
    corrections = corrector.suggest("今天我们聊阿里爸爸的故事")
    assert corrections == []
