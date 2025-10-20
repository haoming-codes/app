from asr_postcorrection import Lexicon, PhoneticMatcher


def test_matcher_corrects_single_entity():
    lexicon = Lexicon.from_strings(["张三"])
    matcher = PhoneticMatcher(lexicon)
    corrected, candidates = matcher.apply_best_corrections("张山", threshold=1.4)
    assert corrected == "张三"
    assert candidates
    assert candidates[0].replacement == "张三"
