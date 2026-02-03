def test_version_present_and_semver():
    import re
    import firepype

    assert hasattr(firepype, "__version__")
    assert re.match(r"^\d+\.\d+\.\d+([a-zA-Z0-9\.-]*)?$", firepype.__version__)
