from ragasr import ASRCorrector, CorrectionConfig, DistanceConfig, KnowledgeEntry


def test_corrector_replaces_known_entities():
    kb = [KnowledgeEntry("美国"), KnowledgeEntry("Mini Map")]
    config = CorrectionConfig(distance=DistanceConfig(threshold=0.12))
    corrector = ASRCorrector(kb, config)
    corrected, candidates = corrector.correct("成哥 去 mango 看 minivan")
    assert "美国" in corrected
    assert "Mini Map" in corrected
    surfaces = {c.entry.surface for c in candidates}
    assert {"美国", "Mini Map"}.issubset(surfaces)
    assert all(c.distance <= config.distance.threshold for c in candidates)


def test_canonical_replacement():
    kb = [KnowledgeEntry("AG.AL", canonical="A.G.A.L.")]
    config = CorrectionConfig(distance=DistanceConfig(threshold=0.2))
    corrector = ASRCorrector(kb, config)
    corrected, _ = corrector.correct("agal")
    assert corrected == "A.G.A.L."
