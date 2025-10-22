from phonetic_rag.config import DistanceAggregationConfig, PhoneticPipelineConfig
from phonetic_rag.corrector import PhoneticCorrector


def test_corrector_finds_closest_window():
    aggregation = DistanceAggregationConfig(lambda_segment=0.7, lambda_tone=0.2, lambda_stress=0.1)
    config = PhoneticPipelineConfig(aggregation=aggregation, window_threshold=0.35, max_candidates=2)
    corrector = PhoneticCorrector(['Mini Map'], config=config)
    matches = corrector.correct('open the minivan please')
    assert matches
    best = matches[0]
    assert best.replacement == 'Mini Map'
    assert best.start == 9  # span of 'minivan'


def test_corrector_handles_no_match():
    config = PhoneticPipelineConfig(window_threshold=0.2)
    corrector = PhoneticCorrector(['Mini Map'], config=config)
    matches = corrector.correct('totally unrelated words')
    assert matches == []
