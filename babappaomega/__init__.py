"""Legacy compatibility namespace for renamed BABAPPAi package."""

from warnings import warn

from babappai import *  # noqa: F401,F403
from babappai import __version__

warn(
    "'babappaomega' is a legacy compatibility namespace. Use 'babappai' instead. "
    "BABAPPAi is the renamed continuation of BABAPPAΩ.",
    DeprecationWarning,
    stacklevel=2,
)
