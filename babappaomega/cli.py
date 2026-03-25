"""Legacy CLI entry point shim."""

import sys

from babappai.cli import main as _main


def main(argv=None):
    print(
        "[DEPRECATION] 'babappaomega' CLI is legacy. Use 'babappai'. "
        "BABAPPAi is the renamed continuation of BABAPPAΩ.",
        file=sys.stderr,
    )
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
