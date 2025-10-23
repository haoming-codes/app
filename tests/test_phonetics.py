from rag_asr_corrector.config import DistanceConfig
from rag_asr_corrector.phonetics import PhonemizerService, ipa_tokens


def test_tokenize_mixed_bilingual_text():
    service = PhonemizerService(DistanceConfig())
    tokens = service.tokenize('美国Mini Map')
    assert [token.text for token in tokens] == ['美', '国', 'Mini', 'Map']
    assert [token.language for token in tokens] == ['cmn', 'cmn', 'en', 'en']


def test_acronym_is_spelled_out():
    tokens = ipa_tokens('AG.AL')
    assert len(tokens) == 1
    ipa = tokens[0].ipa
    assert ' ' in ipa  # spelled out as separate letters
    assert ipa.replace(' ', '').startswith('ɐ') or ipa.replace(' ', '').startswith('e')
