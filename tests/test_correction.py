from asr_corrector.corrector import NameCorrectionPipeline
from asr_corrector.entities import EntityLexicon, Entity
from asr_corrector.matcher import LevenshteinDistance
from asr_corrector.phonetics import MappingEncoder


class DummyEncoder(MappingEncoder):
    def __init__(self):
        mapping = {
            "马": "ma3",
            "斯": "si1",
            "克": "ke4",
            "科": "ke1",
            "张": "zhang1",
            "三": "san1",
            "丰": "feng1",
            "峰": "feng1",
        }
        super().__init__(mapping)


def test_pipeline_fixes_common_errors():
    encoder = DummyEncoder()
    entities = [
        Entity.from_surface("马斯克", encoder),
        Entity.from_surface("张三丰", encoder),
    ]
    lexicon = EntityLexicon(entities)
    pipeline = NameCorrectionPipeline(
        lexicon=lexicon,
        encoder=encoder,
        distance_metric=LevenshteinDistance(),
        distance_threshold=2.0,
    )

    text = "今天我们采访了马斯科和张三峰"
    result = pipeline.correct(text)

    assert result.corrected_text == "今天我们采访了马斯克和张三丰"
    assert len(result.corrections) == 2
    assert result.corrections[0].original == "马斯科"
    assert result.corrections[0].replacement == "马斯克"
