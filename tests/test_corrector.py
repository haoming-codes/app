from asr_corrector import NameEntity, PhoneticCorrector


def test_basic_correction():
    entities = [NameEntity("阿里巴巴"), NameEntity("华为")]
    corrector = PhoneticCorrector(entities, threshold=0.35)

    transcript = "我们参观了阿里八八的总部"
    result = corrector.correct(transcript)

    assert result.corrected == "我们参观了阿里巴巴的总部"
    assert result.matches
    assert result.matches[0].entity.canonical == "阿里巴巴"
    assert result.matches[0].original == "阿里八八"


def test_no_false_positive_when_distance_high():
    entities = [NameEntity("阿里巴巴")]
    corrector = PhoneticCorrector(entities, threshold=0.2)

    transcript = "我们参观了阿里八八的总部"
    result = corrector.correct(transcript)

    assert result.corrected == transcript
    assert not result.matches
