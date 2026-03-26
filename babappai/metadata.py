"""Canonical metadata for BABAPPAi and legacy frozen assets."""

PACKAGE_NAME = "babappai"
SOFTWARE_NAME = "BABAPPAi"
SOFTWARE_VERSION = "2.1.0"
SOFTWARE_DOI = "10.5281/zenodo.18520163"

LEGACY_CODEBASE_NAME = "BABAPPAΩ"
LEGACY_MODEL_NAME = "BABAPPAΩ frozen model"
MODEL_TAG = "legacy_frozen"
MODEL_FILE_NAME = "babappaomega.pt"
MODEL_DOI = "10.5281/zenodo.18195869"
MODEL_URL = "https://zenodo.org/records/18195869/files/babappaomega.pt?download=1"
MODEL_SHA256 = "657a662563af31304abcb208fc903d2770a9184632a9bab2095db4c538fed8eb"
MODEL_COMPATIBILITY_NOTE = (
    "BABAPPAi is the renamed continuation of the BABAPPAΩ codebase. "
    "The cached frozen model is a legacy BABAPPAΩ asset used for "
    "backward-compatible inference until BABAPPAi-specific weights are released."
)
