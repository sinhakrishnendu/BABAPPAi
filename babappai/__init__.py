"""BABAPPAi package metadata exports."""

from babappai.metadata import (
    LEGACY_CODEBASE_NAME,
    MODEL_COMPATIBILITY_NOTE,
    MODEL_LINEAGE_NAME,
    MODEL_NAME,
    MODEL_ROLE,
    PACKAGE_NAME,
    SOFTWARE_NAME,
    SOFTWARE_VERSION,
)

__version__ = SOFTWARE_VERSION

__all__ = [
    "__version__",
    "PACKAGE_NAME",
    "SOFTWARE_NAME",
    "LEGACY_CODEBASE_NAME",
    "MODEL_LINEAGE_NAME",
    "MODEL_NAME",
    "MODEL_ROLE",
    "MODEL_COMPATIBILITY_NOTE",
]
