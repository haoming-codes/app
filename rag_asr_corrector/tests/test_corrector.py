from rag_asr_corrector.config import CorrectionConfig, DistanceConfig
from rag_asr_corrector.corrector import PhoneticCorrector
from rag_asr_corrector.knowledge_base import KnowledgeBase


def make_corrector(threshold: float = 0.25) -> PhoneticCorrector:
    kb = KnowledgeBase()
    kb.add("美国")
    kb.add("Mini Map")
    config = CorrectionConfig(distance=DistanceConfig(stress_weight=0.1), threshold=threshold)
    return PhoneticCorrector(kb, config=config)


def test_suggestions_ranked_by_distance():
    corrector = make_corrector(0.3)
    text = "mango mini map"
    suggestions = corrector.suggestions(text)
    assert suggestions
    assert suggestions[0].replacement in {"美国", "Mini Map"}


def test_apply_replaces_best_matches():
    corrector = make_corrector(0.3)
    text = "mango minivan"
    corrected, applied = corrector.apply(text)
    assert "美国" in corrected or "Mini Map" in corrected
    assert applied
