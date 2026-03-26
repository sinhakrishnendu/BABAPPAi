from babappai.identifiability import (
    eii01_from_eiiz,
    identifiability_extent,
    interpret_identifiability,
)


def test_eii01_bounded():
    assert 0.0 <= eii01_from_eiiz(-20.0) <= 1.0
    assert 0.0 <= eii01_from_eiiz(0.0) <= 1.0
    assert 0.0 <= eii01_from_eiiz(20.0) <= 1.0


def test_regime_boundaries():
    assert identifiability_extent(0.00) == "not_identifiable"
    assert identifiability_extent(0.29) == "not_identifiable"
    assert identifiability_extent(0.30) == "weak_or_ambiguous"
    assert identifiability_extent(0.69) == "weak_or_ambiguous"
    assert identifiability_extent(0.70) == "identifiable"
    assert identifiability_extent(0.89) == "identifiable"
    assert identifiability_extent(0.90) == "strongly_identifiable"


def test_interpret_identifiability_fields():
    result = interpret_identifiability(1.7)
    assert set(result.keys()) == {"EII_01", "identifiable_bool", "identifiability_extent", "eii_band_descriptive_only"}
    assert isinstance(result["identifiable_bool"], bool)
