from rag_asr_corrector.config import DistanceConfig, DistanceLambdas, SegmentWeights
from rag_asr_corrector.distance import PronunciationDistance


def test_identical_strings_have_zero_distance():
    distance = PronunciationDistance()
    breakdown = distance.distance('Mini Map', 'Mini Map')
    assert breakdown.total == 0.0
    assert breakdown.segment_components['phonetic_edit'] == 0.0


def test_different_strings_increase_distance():
    distance = PronunciationDistance()
    same = distance.distance('Mini Map', 'Mini Map').total
    different = distance.distance('Mini Map', 'minivan').total
    assert different > same


def test_custom_weights_affect_segment_component():
    config = DistanceConfig(
        segment_weights=SegmentWeights(phonetic_edit=1.0, aline=0.0, feature=0.0),
        lambdas=DistanceLambdas(segment=1.0, tone=0.0, stress=0.0),
    )
    distance = PronunciationDistance(config)
    breakdown = distance.distance('Mini Map', 'minivan')
    assert breakdown.total == breakdown.segment_components['phonetic_edit']
