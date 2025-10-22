from asr_corrector.matcher import KnowledgeBaseMatcher, MatchingConfig


def test_matcher_highlights_best_candidate():
    config = MatchingConfig(decision_threshold=0.05)
    matcher = KnowledgeBaseMatcher(["美国", "成哥", "Mini Map"], config=config)
    results = matcher.match("今天mango很好吃还有Mini Map的问题")
    assert any(r.entry.surface == "Mini Map" for r in results)
    top = results[0]
    assert top.entry.surface == "Mini Map"
    assert top.window_text == "Mini Map"


def test_matcher_respects_threshold():
    config = MatchingConfig(decision_threshold=0.01)
    matcher = KnowledgeBaseMatcher(["美国"], config=config)
    results = matcher.match("今天mango很好吃还有Mini Map的问题")
    assert all(r.distance.combined <= config.decision_threshold for r in results)
