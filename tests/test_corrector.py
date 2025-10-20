from asr_name_corrector.lexicon import EntityLexicon
from asr_name_corrector.corrector import EntityCorrector


def test_corrector_fixes_near_homophone():
    lexicon = EntityLexicon.from_strings(["张三", "李四"])
    corrector = EntityCorrector(lexicon, threshold=0.4, lambda_weight=0.2)

    # "章三" is a common substitution for "张三"
    result = corrector.correct("昨天章三来了")
    assert result.corrected_text == "昨天张三来了"
    assert result.replacements


def test_corrector_keeps_non_matches():
    lexicon = EntityLexicon.from_strings(["王五"])
    corrector = EntityCorrector(lexicon, threshold=0.3)

    result = corrector.correct("今天我们去公园")
    assert result.corrected_text == "今天我们去公园"
    assert not result.replacements
