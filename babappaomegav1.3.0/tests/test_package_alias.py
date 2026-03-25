import warnings


def test_babappai_import():
    import babappai

    assert hasattr(babappai, "__version__")


def test_legacy_babappaomega_namespace():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        import babappaomega

    assert hasattr(babappaomega, "__version__")


def test_legacy_cli_alias_runs_help():
    from babappaomega.cli import main as legacy_main

    code = legacy_main(["--help"])
    assert code == 0
