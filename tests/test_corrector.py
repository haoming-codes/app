from asr_corrector.lexicon import NameLexicon
from asr_corrector.corrector import RagBasedCorrector
from asr_corrector.phonetics import PhoneticTranscriber


def test_transcriber_handles_basic_names():
    transcriber = PhoneticTranscriber()
    ipa = transcriber.transcribe("张伟")
    assert "˥" in ipa
    assert "wei" in ipa


def test_corrector_applies_best_candidate():
    records = [
        {"canonical": "张伟", "aliases": ["章惟"]},
        {"canonical": "王芳", "aliases": ["王方"]},
    ]
    lexicon = NameLexicon.from_records(records)
    corrector = RagBasedCorrector(lexicon, threshold=0.5)
    text = "今天我遇到了章惟和王方"
    result = corrector.apply(text)
    assert result.text == "今天我遇到了张伟和王芳"
    assert {c.original for c in result.applied} == {"章惟", "王方"}
    assert all(c.similarity > 0.5 for c in result.applied)
