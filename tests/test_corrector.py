from rag_asr_corrector.config import CorrectionConfig, DistanceConfig, DistanceLambdas, SegmentWeights
from rag_asr_corrector.corrector import ASRCorrector


def test_corrector_suggests_known_name():
    config = CorrectionConfig(
        distance=DistanceConfig(
            segment_weights=SegmentWeights(phonetic_edit=0.7, aline=0.3, feature=0.0),
            lambdas=DistanceLambdas(segment=1.0, tone=0.0, stress=0.0),
        ),
        threshold=0.4,
    )
    corrector = ASRCorrector(['成哥'], config)
    suggestions = corrector.suggest('恒哥来了')
    assert suggestions
    assert any(s.candidate == '成哥' and s.original_text == '恒哥' for s in suggestions)


def test_no_suggestion_when_languages_do_not_match():
    corrector = ASRCorrector(['美国'])
    suggestions = corrector.suggest('mango is sweet')
    assert not suggestions
