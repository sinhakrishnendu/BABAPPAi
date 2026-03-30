"""Compatibility namespace for importing BABAPPAi via the historical name."""

from warnings import warn

from babappai import *  # noqa: F401,F403
from babappai import __version__

warn(
    "'babappaomega' is a compatibility alias. Prefer 'babappai' for software commands; "
    "the canonical frozen inference model remains BABAPPAΩ.",
    DeprecationWarning,
    stacklevel=2,
)
