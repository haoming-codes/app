from asr_corrector.phonetics import IPAConverter


def test_ipa_converter_returns_features():
    converter = IPAConverter()
    result = converter.to_ipa("Mini Map")
    assert result.ipa
    assert len(result.phones) == result.feature_vectors.shape[0]
    assert result.feature_vectors.shape[1] > 0


def test_acronym_handling():
    converter = IPAConverter()
    acronym = converter.to_ipa("NASA", is_acronym=True)
    normal = converter.to_ipa("NASA", is_acronym=False)
    assert acronym.ipa != normal.ipa
