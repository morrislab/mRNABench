from mrna_bench.models import MODEL_CATALOG, ModelBehavior


def test_every_model_declares_behaviors():
    """Every model explicitly declares common or version-specific behavior."""
    for model_class in MODEL_CATALOG.values():
        assert (
            "supported_behaviors" in model_class.__dict__
            or "behaviors_for_version" in model_class.__dict__
        )


def test_masked_lm_behaviors_match_retained_heads():
    """Masked-LM behavior is declared only where a head is retained."""
    masked_models = {
        "AIDO.DNA",
        "AIDO.RNA",
        "CodonBERT",
        "DNABERT",
        "DNABERT2",
        "ERNIE-RNA",
        "mRNABERT",
        "mRNA-FM",
        "OmniGenome",
        "Plant-RNAFM",
        "RiNALMo",
        "RNABERT",
        "RNAErnie",
        "RNA-FM",
        "RNA-MSM",
        "SpliceBERT",
        "3UTRBERT",
        "UTR-LM",
    }
    for model_name in masked_models:
        model_class = MODEL_CATALOG[model_name]
        for version in model_class.valid_versions:
            behaviors = model_class.behaviors_for_version(version)
            assert ModelBehavior.PSEUDO_LIKELIHOOD in behaviors

    generanno = MODEL_CATALOG["GENERanno"]
    assert ModelBehavior.PSEUDO_LIKELIHOOD in generanno.behaviors_for_version(
        "eukaryote-0.5b-base"
    )
    assert (
        ModelBehavior.PSEUDO_LIKELIHOOD
        not in generanno.behaviors_for_version(
            "eukaryote-1.2b-cds-annotator-preview"
        )
    )
    dnabert_s = MODEL_CATALOG["DNABERT-S"]
    for version in dnabert_s.valid_versions:
        assert (
            ModelBehavior.PSEUDO_LIKELIHOOD
            not in dnabert_s.behaviors_for_version(version)
        )
