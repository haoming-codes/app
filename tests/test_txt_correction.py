from pathlib import Path

import pytest

from asr_corrector import (
    ASRCorrector,
    CorrectionConfig,
    KnowledgeBase,
    KnowledgeEntry,
    MetricConfig,
    ToneConfig,
)


@pytest.mark.parametrize("threshold", [1.2])
def test_correction_from_txt_inputs(tmp_path: Path, threshold: float) -> None:
    """Read transcripts and entities from text files and correct mis-transcriptions."""
    transcripts = [
        "我们今天学习沫子的思想",
        "我们参观了清华大學校园",
    ]
    transcript_file = tmp_path / "transcripts.txt"
    transcript_file.write_text("\n".join(transcripts), encoding="utf-8")

    entities = ["墨子", "清华大学"]
    entity_file = tmp_path / "entities.txt"
    entity_file.write_text("\n".join(entities), encoding="utf-8")

    knowledge_entries = [
        KnowledgeEntry(line.strip())
        for line in entity_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    knowledge_base = KnowledgeBase(knowledge_entries)

    config = CorrectionConfig(
        metrics=[MetricConfig(name="panphon", weight=1.0)],
        tone=ToneConfig(weight=1.0, default_cost=1.0),
        threshold=threshold,
        window_sizes=[2, 4],
    )
    corrector = ASRCorrector(knowledge_base, config)

    corrected_sentences = []
    matched_pairs = []
    for line in transcript_file.read_text(encoding="utf-8").splitlines():
        result = corrector.correct(line)
        corrected_sentences.append(result.text)
        matched_pairs.extend((repl.original, repl.replacement) for repl in result.replacements)

    assert corrected_sentences == [
        "我们今天学习墨子的思想",
        "我们参观了清华大学校园",
    ]
    assert ("沫子", "墨子") in matched_pairs
    assert ("清华大學", "清华大学") in matched_pairs
    assert all(pair[0] != pair[1] for pair in matched_pairs)
