from asr_correction.matcher import KnowledgeBaseMatcher


def test_matcher_recovers_known_entities():
    matcher = KnowledgeBaseMatcher(['美国', '成哥', 'Mini Map'])
    text = '我们讨论了mango, 然后提到了Mini Map'
    matches = matcher.find_matches(text, threshold=0.3, max_window_size=3)
    assert any(match.entry.original == '美国' for match in matches)
    assert any(match.entry.original == 'Mini Map' and match.breakdown.overall == 0.0 for match in matches)


def test_acronym_entries_are_spelled_out():
    matcher = KnowledgeBaseMatcher(['AG.AL'])
    assert matcher.entries[0].normalized == 'A G A L'
