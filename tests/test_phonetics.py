from asr_name_corrector.phonetics import MandarinTranscriber


def test_transcriber_returns_tone_marks():
    transcriber = MandarinTranscriber()
    transcription = transcriber.transcribe("张三")
    ipa = transcription.ipa
    assert "˥" in ipa or "˧" in ipa
    assert len(transcription.ipa_syllables) == len("张三")
