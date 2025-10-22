from phonetic_correction.phonetics import MultilingualPhonemizer


def test_transcribe_mixed_text_produces_expected_sequences():
    phonemizer = MultilingualPhonemizer()
    transcription = phonemizer.transcribe("美国 Mini Map")

    assert transcription.ipa.startswith("meik")
    assert transcription.tone_sequence == [3, 2]
    assert transcription.tone_unit_count == 2
    assert transcription.stress_sequence == [1, 1]
    assert transcription.stress_unit_count == 2
    assert len(transcription.feature_vectors) == len(transcription.segments)
    assert len(transcription.feature_vectors) > 0
    feature_dim = transcription.feature_vectors[0].shape[0]
    assert all(vector.shape == (feature_dim,) for vector in transcription.feature_vectors)
