from asr_corrector import PhoneticTranscriber


def test_transcribe_acronym_produces_letter_ipa():
    transcriber = PhoneticTranscriber()
    rep = transcriber.transcribe("AG.AL")
    assert rep.ipa == "eɪdʒiːeɪɛl"
    assert rep.tones == []
    assert rep.stresses == []
    assert len(rep.features) > 0


def test_transcribe_tracks_tone_and_stress():
    transcriber = PhoneticTranscriber()
    zh_rep = transcriber.transcribe("美国")
    assert zh_rep.tones == [2]

    en_rep = transcriber.transcribe("banana")
    assert en_rep.stresses == [2]
